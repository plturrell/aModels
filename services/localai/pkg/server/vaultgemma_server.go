// Package server provides the HTTP server implementation for the LocalAI VaultGemma service.
// It handles chat completions, embeddings, streaming, function calling, and other
// OpenAI-compatible API endpoints. The server supports multiple backends including
// safetensors (CPU), GGUF (CPU/GPU), and HuggingFace transformers (GPU).
package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/gpu"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/hanapool"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/gguf"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/transformers"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/vision"
	"golang.org/x/time/rate"
	"os"
)

var (
	defaultTopP = ai.DefaultTopP
	defaultTopK = ai.DefaultTopK
)

type hanaCacheStore interface {
	GenerateCacheKey(prompt, model, domain string, temperature float64, maxTokens int, topP float64, topK int) string
	Get(ctx context.Context, key string) (*storage.CacheEntry, error)
	Set(ctx context.Context, entry *storage.CacheEntry) error
}

type semanticCacheStore interface {
	GenerateCacheKey(prompt, model, domain string, temperature float64, maxTokens int, topP float64, topK int) string
	GenerateSemanticHash(prompt string) string
	FindSemanticSimilar(ctx context.Context, prompt, model, domain string, threshold float64, limit int) ([]*storage.SemanticCacheEntry, error)
	Set(ctx context.Context, entry *storage.SemanticCacheEntry) error
	CreateTables(ctx context.Context) error
	VectorSearchEnabled() bool
}

func resolveSampling(topP float64, topK int, domainConfig *domain.DomainConfig) (float64, int) {
	if domainConfig != nil {
		if topP <= 0 && domainConfig.TopP > 0 {
			topP = float64(domainConfig.TopP)
		}
		if topK <= 0 && domainConfig.TopK > 0 {
			topK = domainConfig.TopK
		}
	}
	if topP <= 0 {
		topP = defaultTopP
	}
	if topP > 1 {
		topP = 1
	}
	if topK <= 0 {
		topK = defaultTopK
	}
	return topP, topK
}

type VaultGemmaServer struct {
	models             map[string]*ai.VaultGemma
	ggufModels         map[string]*gguf.Model
	transformerClients map[string]*transformers.Client
	domainManager      *domain.DomainManager
	limiter            *rate.Limiter
	inferenceEngine    *inference.InferenceEngine
	enhancedEngine     *inference.EnhancedInferenceEngine
	hanaPool           *hanapool.Pool
	hanaLogger         *storage.HANALogger
	hanaCache          hanaCacheStore
	semanticCache      semanticCacheStore
	enhancedLogging    *EnhancedLogging
	requestCount       int64
	mu                 sync.RWMutex
	startTime          time.Time
	version            string
	// New enhanced features
	tokenCounter     *TokenCounter
	functionRegistry *FunctionRegistry
	retryConfig      *RetryConfig
	ocrServices      map[string]*vision.DeepSeekOCRService
	agentCatalogMu   sync.RWMutex
	agentCatalog     *AgentCatalog
	catalogUpdatedAt time.Time
	flightAddr       string
	gpuRouter        *gpu.GPURouter
	// Performance features
	modelCache       *ModelCache
	batchProcessor   *BatchProcessor
	profiler         *Profiler
	// Postgres integration (Phase 1)
	postgresLogger   *storage.PostgresInferenceLogger
	postgresCache    *storage.PostgresCacheStore
}

// ModelRegistry exposes configuration metadata for orchestration layer discovery
type ModelRegistry struct {
	Domains             map[string]*domain.DomainConfig `json:"domains"`
	AgentCatalog        *AgentCatalog                   `json:"agent_catalog,omitempty"`
	AgentCatalogUpdated string                          `json:"agent_catalog_updated_at,omitempty"`
}

// SetFlightAddr records the Flight endpoint the server can refresh from.
func (s *VaultGemmaServer) SetFlightAddr(addr string) {
	s.agentCatalogMu.Lock()
	defer s.agentCatalogMu.Unlock()
	s.flightAddr = strings.TrimSpace(addr)
}

// UpdateAgentCatalog replaces the cached Agent SDK catalog.
func (s *VaultGemmaServer) UpdateAgentCatalog(cat AgentCatalog) {
	cat.Normalize()
	s.agentCatalogMu.Lock()
	defer s.agentCatalogMu.Unlock()
	s.agentCatalog = cat.Clone()
	s.catalogUpdatedAt = time.Now().UTC()
}

func (s *VaultGemmaServer) agentCatalogSnapshot() (*AgentCatalog, time.Time) {
	s.agentCatalogMu.RLock()
	defer s.agentCatalogMu.RUnlock()
	if s.agentCatalog == nil {
		return nil, time.Time{}
	}
	return s.agentCatalog.Clone(), s.catalogUpdatedAt
}

// AgentCatalogSnapshot exposes the cached catalog for callers outside the server package.
func (s *VaultGemmaServer) AgentCatalogSnapshot() (*AgentCatalog, time.Time) {
	return s.agentCatalogSnapshot()
}

// FlightAddr reports the configured Arrow Flight endpoint.
func (s *VaultGemmaServer) FlightAddr() string {
	s.agentCatalogMu.RLock()
	defer s.agentCatalogMu.RUnlock()
	return s.flightAddr
}

func (s *VaultGemmaServer) enrichPromptWithAgentCatalog(prompt string) string {
	catalog, _ := s.agentCatalogSnapshot()
	if catalog == nil || (len(catalog.Suites) == 0 && len(catalog.Tools) == 0) {
		return prompt
	}

	var builder strings.Builder
	builder.WriteString(prompt)
	builder.WriteString("\n\nAgent SDK context:\n")
	for _, suite := range catalog.Suites {
		builder.WriteString(fmt.Sprintf("- Suite %s", suite.Name))
		if suite.Implementation != "" {
			builder.WriteString(fmt.Sprintf(" (implementation: %s", suite.Implementation))
			if suite.Version != "" {
				builder.WriteString(fmt.Sprintf(" v%s", suite.Version))
			}
			builder.WriteString(")")
		}
		if len(suite.ToolNames) > 0 {
			builder.WriteString(fmt.Sprintf(" tools: %s", strings.Join(suite.ToolNames, ", ")))
		}
		if !suite.AttachedAt.IsZero() {
			builder.WriteString(fmt.Sprintf(" attached_at=%s", suite.AttachedAt.UTC().Format(time.RFC3339)))
		}
		builder.WriteString("\n")
	}
	if len(catalog.Tools) > 0 {
		builder.WriteString("Available tools:\n")
		for _, tool := range catalog.Tools {
			if strings.TrimSpace(tool.Description) != "" {
				builder.WriteString(fmt.Sprintf("  - %s: %s\n", tool.Name, tool.Description))
			} else {
				builder.WriteString(fmt.Sprintf("  - %s\n", tool.Name))
			}
		}
	}
	return builder.String()
}

// HandleChat handles chat completion requests using refactored helper functions
func (s *VaultGemmaServer) HandleChat(w http.ResponseWriter, r *http.Request) {
	// Create context with timeout
	ctx, cancel := context.WithTimeout(r.Context(), RequestTimeoutDefault)
	defer cancel()

	// Check if request batching is enabled
	if s.batchProcessor != nil && os.Getenv("ENABLE_REQUEST_BATCHING") == "1" {
		s.handleChatWithBatching(w, r, ctx)
		return
	}

	// Track request
	s.mu.Lock()
	s.requestCount++
	s.mu.Unlock()

	start := time.Now()

	// Validate and decode request
	req, err := validateChatRequest(r)
	if err != nil {
		handleChatError(w, err, http.StatusBadRequest)
		return
	}

	// Check context deadline
	select {
	case <-ctx.Done():
		handleChatError(w, ErrTimeout, http.StatusRequestTimeout)
		return
	default:
	}

	// Build prompt from messages
	prompt := buildPromptFromMessages(req.Messages)
	prompt = s.enrichPromptWithAgentCatalog(prompt)

	// Detect or use specified domain
	domain := req.Model
	if domain == DomainAuto || domain == "" {
		domain = s.domainManager.DetectDomain(prompt, req.Domains)
		log.Printf("Auto-detected domain: %s", domain)
	}

	// Retrieve domain configuration
	domainConfig, _ := s.domainManager.GetDomainConfig(domain)
	preferredBackend := pickPreferredBackend()

	// Resolve model with fallback
	model, modelKey, fallbackUsed, fallbackKey, err := s.resolveModelForDomain(ctx, domain, domainConfig, preferredBackend)
	if err != nil {
		handleChatError(w, err, http.StatusInternalServerError)
		return
	}

	// Update domain if fallback changed it
	if fallbackUsed && fallbackKey != "" {
		domain = fallbackKey
		domainConfig, _ = s.domainManager.GetDomainConfig(domain)
	}

	// Use domain-specific parameters
	maxTokens := req.MaxTokens
	if maxTokens == 0 && domainConfig != nil {
		maxTokens = domainConfig.MaxTokens
	}
	if maxTokens == 0 {
		maxTokens = DefaultMaxTokens
	}

	topP, topK := resolveSampling(req.TopP, req.TopK, domainConfig)

	// Generate request ID for tracking
	requestID := fmt.Sprintf("req_%d", time.Now().UnixNano())

	// Extract user and session info from headers
	userID := r.Header.Get(HeaderUserID)
	if userID == "" {
		userID = AnonymousUserID
	}
	sessionID := r.Header.Get(HeaderSessionID)
	if sessionID == "" {
		sessionID = DefaultSessionID
	}
	
	// Extract workflow context from headers (Phase 1)
	workflowID := r.Header.Get("X-Workflow-ID")
	if workflowID == "" {
		// Try to get from request body if available
		if reqBody, ok := r.Context().Value("request_body").(map[string]interface{}); ok {
			if wfID, ok := reqBody["workflow_id"].(string); ok {
				workflowID = wfID
			}
		}
	}

	// Log request start
	if s.enhancedLogging != nil {
		s.enhancedLogging.LogRequestStart(ctx, requestID, modelKey, domain, prompt, userID, sessionID)
	}

	// Process chat request
	result, err := s.processChatRequest(ctx, req, domain, domainConfig, model, modelKey, prompt, maxTokens, topP, topK, requestID, userID, sessionID)
	if err != nil {
		// Record error in profiler
		if s.profiler != nil {
			s.profiler.RecordError()
		}
		handleChatError(w, err, http.StatusBadGateway)
		return
	}

	duration := time.Since(start)
	log.Printf("Chat request processed in %.2fms for domain: %s", duration.Seconds()*1000, domain)

	// Record profiling metrics
	if s.profiler != nil {
		s.profiler.RecordRequest(duration)
	}

	// Log inference to Postgres (Phase 1) - non-blocking
	if s.postgresLogger != nil {
		go func() {
			logEntry := &storage.InferenceLog{
				Domain:          domain,
				ModelName:       modelKey,
				PromptLength:    len(prompt),
				ResponseLength:  len(result.Content),
				LatencyMS:       int(duration.Milliseconds()),
				TokensGenerated: result.TokensUsed,
				TokensPrompt:    len(prompt) / 4, // Rough estimate
				WorkflowID:      workflowID,
				UserID:          userID,
				CreatedAt:       time.Now(),
			}
			if err := s.postgresLogger.LogInference(context.Background(), logEntry); err != nil {
				log.Printf("⚠️  Failed to log inference to Postgres: %v", err)
			}
		}()
	}

	// Update metadata with final values
	if _, ok := result.Metadata["backend_type"]; !ok {
		result.Metadata["backend_type"] = BackendTypeSafetensors
	}
	result.Metadata["cache_hit"] = result.CacheHit
	if domainConfig != nil {
		result.Metadata["domain_name"] = domainConfig.Name
	}
	if fallbackUsed {
		result.Metadata["fallback_used"] = true
		result.Metadata["fallback_model"] = fallbackKey
		if s.enhancedLogging != nil {
			s.enhancedLogging.LogModelSwitch(ctx, requestID, modelKey, fallbackKey, domain, prompt, "model_unavailable", userID, sessionID)
		}
	} else {
		result.Metadata["fallback_used"] = false
	}

	// Log request completion
	if s.enhancedLogging != nil {
		s.enhancedLogging.LogRequestEnd(ctx, requestID, modelKey, domain, prompt, result.Content, result.TokensUsed, duration.Milliseconds(), req.Temperature, maxTokens, result.CacheHit, result.SemanticHit, userID, sessionID, result.Metadata)
	}

	// Save to cache if not already cached
	if !result.HandledExternally && !result.CacheHit && !result.SemanticHit {
		s.saveToCache(ctx, prompt, modelKey, domain, result.Content, result.TokensUsed, req.Temperature, maxTokens, topP, topK)
	}

	// Build and send response
	resp := buildChatResponse(modelKey, result.Content, result.TokensUsed, prompt, result.Metadata)
	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(resp)
}

// handleChatError handles errors in chat requests with proper error formatting
func handleChatError(w http.ResponseWriter, err error, statusCode int) {
	errorMsg := err.Error()
	errorCode := ErrorCodeInternalError

	// Extract error code if it's a known error
	if errors.Is(err, ErrInvalidRequest) {
		errorCode = ErrorCodeInvalidRequest
	} else if errors.Is(err, ErrModelNotFound) {
		errorCode = ErrorCodeModelNotFound
	} else if errors.Is(err, ErrBackendUnavailable) {
		errorCode = ErrorCodeBackendUnavailable
	} else if errors.Is(err, ErrTimeout) {
		errorCode = ErrorCodeTimeout
	} else if errors.Is(err, ErrInternalError) {
		errorCode = ErrorCodeInternalError
	}

	// Build error response
	errorResponse := map[string]interface{}{
		"error": map[string]interface{}{
			"message": errorMsg,
			"type":    errorCode,
			"code":    errorCode,
		},
	}

	w.Header().Set(HeaderContentType, ContentTypeJSON)
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(errorResponse)
}

// pickPreferredBackend decides which backend to use when not specified in config.
// Order:
// 1) hf-transformers if TRANSFORMERS_BASE_URL is set (GPU or CPU)
// 2) gguf if GGUF_ENABLE=1
// 3) vaultgemma as a safe pure-Go fallback
func pickPreferredBackend() string {
	base := strings.TrimSpace(os.Getenv("TRANSFORMERS_BASE_URL"))
	if base != "" {
		return BackendTypeTransformers
	}
	if strings.EqualFold(strings.TrimSpace(os.Getenv("GGUF_ENABLE")), "1") {
		return BackendTypeGGUF
	}
	return BackendTypeSafetensors
}

func (s *VaultGemmaServer) HandleModels(w http.ResponseWriter, r *http.Request) {
	resp := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":       "vaultgemma",
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "google",
			},
		},
	}

	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(resp)
}

// HandleEmbeddings handles OpenAI-compatible embeddings requests
func (s *VaultGemmaServer) HandleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if r.Header.Get(HeaderContentType) != ContentTypeJSON {
		http.Error(w, "Content-Type must be application/json", http.StatusUnsupportedMediaType)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), RequestTimeoutEmbeddings)
	defer cancel()

	var req struct {
		Model string      `json:"model"`
		Input interface{} `json:"input"` // Can be string or []string
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Get model (use vaultgemma as default) - Phase 4: Use lazy loading
	var model *ai.VaultGemma
	if s.modelCache != nil {
		ctx, cancel := context.WithTimeout(r.Context(), RequestTimeoutEmbeddings)
		defer cancel()
		loadedModel, err := s.modelCache.GetSafetensorModel(ctx, "vaultgemma")
		if err == nil && loadedModel != nil {
			model = loadedModel
		}
	}
	
	// Fallback to pre-loaded model
	if model == nil {
		s.mu.RLock()
		model = s.models["vaultgemma"]
		s.mu.RUnlock()
	}
	
	if model == nil {
		http.Error(w, "Model not available", http.StatusServiceUnavailable)
		return
	}

	// Convert input to []string
	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		inputs = make([]string, len(v))
		for i, item := range v {
			if str, ok := item.(string); ok {
				inputs[i] = str
			} else {
				http.Error(w, "Input must be string or array of strings", http.StatusBadRequest)
				return
			}
		}
	case []string:
		inputs = v
	default:
		http.Error(w, "Input must be string or array of strings", http.StatusBadRequest)
		return
	}

	// Generate embeddings for each input
	type EmbeddingItem struct {
		Object    string    `json:"object"`
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	}

	items := make([]EmbeddingItem, 0, len(inputs))
	for idx, text := range inputs {
		// Simple tokenization (same as inference engine)
		tokens := make([]int, 0, len(text)/4)
		words := s.splitIntoWords(text)
		for _, word := range words {
			token := s.hashString(word) % model.Config.VocabSize
			if token < 0 {
				token = -token
			}
			tokens = append(tokens, token)
		}
		if len(tokens) == 0 {
			tokens = []int{model.Config.BOSTokenID}
		} else {
			tokens = append([]int{model.Config.BOSTokenID}, tokens...)
		}

		// Get embeddings through model forward pass
		hidden := model.embedTokens(tokens)

		// Mean pool the hidden states to get sentence embedding
		hiddenSize := model.Config.HiddenSize
		embedding := make([]float64, hiddenSize)
		if hidden.Rows > 0 {
			for j := 0; j < hiddenSize; j++ {
				sum := 0.0
				for i := 0; i < hidden.Rows; i++ {
					sum += hidden.Data[i*hiddenSize+j]
				}
				embedding[j] = sum / float64(hidden.Rows)
			}
		}

		items = append(items, EmbeddingItem{
			Object:    "embedding",
			Embedding: embedding,
			Index:     idx,
		})
	}

	resp := map[string]interface{}{
		"object": "list",
		"data":   items,
		"model":  req.Model,
		"usage": map[string]int{
			"prompt_tokens": 0, // Could calculate actual tokens
			"total_tokens":  0,
		},
	}

	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(resp)
}

// HandleDomainRegistry exposes domain metadata for orchestration discovery.
func (s *VaultGemmaServer) HandleDomainRegistry(w http.ResponseWriter, r *http.Request) {
	configs := s.domainManager.ListDomainConfigs()
	catalog, updatedAt := s.agentCatalogSnapshot()
	registry := ModelRegistry{Domains: configs}
	if catalog != nil {
		registry.AgentCatalog = catalog
	}
	if !updatedAt.IsZero() {
		registry.AgentCatalogUpdated = updatedAt.Format(time.RFC3339)
	}

	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(registry)
}

func (s *VaultGemmaServer) HandleHealth(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	requestCount := s.requestCount
	uptime := time.Since(s.startTime)
	s.mu.RUnlock()

	// Get memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	health := map[string]interface{}{
		"status":    "ok",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   s.version,
		"uptime":    uptime.String(),
		"models":    s.getModelsInfo(),
		"features": []string{
			"FlashAttention",
			"SwiGLU",
			"RMSNorm",
			"DifferentialPrivacy",
			"RateLimiting",
			"CORS",
			"EnhancedLogging",
			"SemanticCaching",
		},
		"metrics": map[string]interface{}{
			"requests_total": requestCount,
			"memory": map[string]interface{}{
				"alloc_mb":       float64(m.Alloc) / 1024 / 1024,
				"total_alloc_mb": float64(m.TotalAlloc) / 1024 / 1024,
				"sys_mb":         float64(m.Sys) / 1024 / 1024,
				"num_gc":         m.NumGC,
			},
			"goroutines": runtime.NumGoroutine(),
		},
	}

	// Add enhanced logging stats if available
	if s.enhancedLogging != nil {
		stats := s.enhancedLogging.GetStats()
		health["enhanced_logging"] = stats
	}

	// Log health check
	if s.enhancedLogging != nil {
		s.enhancedLogging.LogHealthCheck(r.Context(), "healthy", health)
	}

	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(health)
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func truncateString(input string, limit int) string {
	if limit <= 0 || len(input) <= limit {
		return input
	}
	if limit <= 3 {
		return input[:limit]
	}
	return input[:limit-3] + "..."
}

func decodeImagePayload(raw string) ([]byte, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, errors.New("empty image payload")
	}

	if comma := strings.Index(trimmed, ","); comma >= 0 && strings.Contains(trimmed[:comma], ";base64") {
		trimmed = trimmed[comma+1:]
	}

	data, err := base64.StdEncoding.DecodeString(trimmed)
	if err != nil {
		return nil, fmt.Errorf("decode base64 image: %w", err)
	}
	return data, nil
}

// rateLimitMiddleware applies rate limiting to requests
func (s *VaultGemmaServer) RateLimitMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.limiter.Allow() {
			log.Printf("Rate limit exceeded from %s", r.RemoteAddr)
			http.Error(w, "Rate limit exceeded. Please try again later.", http.StatusTooManyRequests)
			return
		}
		next.ServeHTTP(w, r)
	}
}

// enableCORS adds CORS headers to responses
func EnableCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	}
}

// getModelsInfo returns information about all loaded models
func (s *VaultGemmaServer) getModelsInfo() []map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	models := make([]map[string]interface{}, 0, len(s.models))
	for domain, model := range s.models {
		config, _ := s.domainManager.GetDomainConfig(domain)

		modelInfo := map[string]interface{}{
			"domain":          domain,
			"layers":          model.Config.NumLayers,
			"hidden_size":     model.Config.HiddenSize,
			"vocab_size":      model.Config.VocabSize,
			"attention_heads": model.Config.NumHeads,
			"implementation":  "pure-go",
		}

		if config != nil {
			modelInfo["name"] = config.Name
			modelInfo["max_tokens"] = config.MaxTokens
			modelInfo["temperature"] = config.Temperature
			modelInfo["tags"] = config.DomainTags
			modelInfo["fallback_model"] = config.FallbackModel
		}

		models = append(models, modelInfo)
	}

	return models
}

// handleListDomains lists all available domains
func (s *VaultGemmaServer) HandleListDomains(w http.ResponseWriter, r *http.Request) {
	domains := make([]map[string]interface{}, 0)

	// Include all domains from domain manager, not just loaded models
	allDomains := s.domainManager.ListDomainConfigs()
	for domainID, config := range allDomains {
		domainInfo := map[string]interface{}{
			"id":     domainID,
			"loaded": false,
		}

		if config != nil {
			domainInfo["name"] = config.Name
			domainInfo["agent_id"] = config.AgentID
			domainInfo["keywords"] = config.Keywords
			domainInfo["tags"] = config.DomainTags
			domainInfo["layer"] = config.Layer
			domainInfo["team"] = config.Team
			domainInfo["max_tokens"] = config.MaxTokens
			domainInfo["fallback_model"] = config.FallbackModel

			// Include full config for domain detection
			domainInfo["config"] = map[string]interface{}{
				"agent_id": config.AgentID,
				"keywords": config.Keywords,
				"tags":     config.DomainTags,
				"layer":    config.Layer,
				"team":     config.Team,
			}
		}

		// Check if model is loaded
		if _, ok := s.models[domainID]; ok {
			domainInfo["loaded"] = true
			if model, ok := s.models[domainID]; ok {
				domainInfo["layers"] = model.Config.NumLayers
				domainInfo["hidden_size"] = model.Config.HiddenSize
			}
		} else if _, ok := s.ggufModels[domainID]; ok {
			domainInfo["loaded"] = true
		} else if _, ok := s.transformerClients[domainID]; ok {
			domainInfo["loaded"] = true
		}

		domains = append(domains, domainInfo)
	}

	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   domains,
	})
}

// HandleMetrics provides Prometheus-style metrics endpoint with enhanced profiling
func (s *VaultGemmaServer) HandleMetrics(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	requestCount := s.requestCount
	uptime := time.Since(s.startTime)
	s.mu.RUnlock()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Simple text-based metrics format
	w.Header().Set("Content-Type", "text/plain")
	fmt.Fprintf(w, "# HELP vaultgemma_requests_total Total number of requests\n")
	fmt.Fprintf(w, "# TYPE vaultgemma_requests_total counter\n")
	fmt.Fprintf(w, "vaultgemma_requests_total %d\n\n", requestCount)

	fmt.Fprintf(w, "# HELP vaultgemma_uptime_seconds Server uptime in seconds\n")
	fmt.Fprintf(w, "# TYPE vaultgemma_uptime_seconds gauge\n")
	fmt.Fprintf(w, "vaultgemma_uptime_seconds %.2f\n", uptime.Seconds())

	// Add profiling stats if available
	if s.profiler != nil {
		stats := s.profiler.GetStats()
		if latency, ok := stats["latency"].(map[string]interface{}); ok {
			if avg, ok := latency["avg_ms"].(int64); ok {
				fmt.Fprintf(w, "# HELP vaultgemma_request_latency_avg_ms Average request latency in milliseconds\n")
				fmt.Fprintf(w, "# TYPE vaultgemma_request_latency_avg_ms gauge\n")
				fmt.Fprintf(w, "vaultgemma_request_latency_avg_ms %d\n", avg)
			}
			if p95, ok := latency["p95_ms"].(int64); ok {
				fmt.Fprintf(w, "# HELP vaultgemma_request_latency_p95_ms 95th percentile request latency in milliseconds\n")
				fmt.Fprintf(w, "# TYPE vaultgemma_request_latency_p95_ms gauge\n")
				fmt.Fprintf(w, "vaultgemma_request_latency_p95_ms %d\n", p95)
			}
			if p99, ok := latency["p99_ms"].(int64); ok {
				fmt.Fprintf(w, "# HELP vaultgemma_request_latency_p99_ms 99th percentile request latency in milliseconds\n")
				fmt.Fprintf(w, "# TYPE vaultgemma_request_latency_p99_ms gauge\n")
				fmt.Fprintf(w, "vaultgemma_request_latency_p99_ms %d\n", p99)
			}
		}
		if errorRate, ok := stats["error_rate"].(float64); ok {
			fmt.Fprintf(w, "# HELP vaultgemma_error_rate Error rate (0.0 to 1.0)\n")
			fmt.Fprintf(w, "# TYPE vaultgemma_error_rate gauge\n")
			fmt.Fprintf(w, "vaultgemma_error_rate %.4f\n", errorRate)
		}
	}

	// Add model cache stats if available
	if s.modelCache != nil {
		cacheStats := s.modelCache.GetStats()
		if safetensorCount, ok := cacheStats["safetensor_models"].(int); ok {
			fmt.Fprintf(w, "# HELP vaultgemma_safetensor_models_loaded Number of loaded safetensors models\n")
			fmt.Fprintf(w, "# TYPE vaultgemma_safetensor_models_loaded gauge\n")
			fmt.Fprintf(w, "vaultgemma_safetensor_models_loaded %d\n", safetensorCount)
		}
		if ggufCount, ok := cacheStats["gguf_models"].(int); ok {
			fmt.Fprintf(w, "# HELP vaultgemma_gguf_models_loaded Number of loaded GGUF models\n")
			fmt.Fprintf(w, "# TYPE vaultgemma_gguf_models_loaded gauge\n")
			fmt.Fprintf(w, "vaultgemma_gguf_models_loaded %d\n", ggufCount)
		}
		if currentMB, ok := cacheStats["current_memory_mb"].(int64); ok {
			fmt.Fprintf(w, "# HELP vaultgemma_model_cache_memory_mb Current memory usage in MB\n")
			fmt.Fprintf(w, "# TYPE vaultgemma_model_cache_memory_mb gauge\n")
			fmt.Fprintf(w, "vaultgemma_model_cache_memory_mb %d\n", currentMB)
		}
		if avgLoadingTime, ok := cacheStats["avg_loading_time_s"].(float64); ok {
			fmt.Fprintf(w, "# HELP vaultgemma_model_avg_loading_time_seconds Average model loading time in seconds\n")
			fmt.Fprintf(w, "# TYPE vaultgemma_model_avg_loading_time_seconds gauge\n")
			fmt.Fprintf(w, "vaultgemma_model_avg_loading_time_seconds %.3f\n", avgLoadingTime)
		}
		if maxLoadingTime, ok := cacheStats["max_loading_time_s"].(float64); ok {
			fmt.Fprintf(w, "# HELP vaultgemma_model_max_loading_time_seconds Maximum model loading time in seconds\n")
			fmt.Fprintf(w, "# TYPE vaultgemma_model_max_loading_time_seconds gauge\n")
			fmt.Fprintf(w, "vaultgemma_model_max_loading_time_seconds %.3f\n", maxLoadingTime)
		}
		if totalLoaded, ok := cacheStats["total_models_loaded"].(int); ok {
			fmt.Fprintf(w, "# HELP vaultgemma_models_loaded_total Total number of models loaded\n")
			fmt.Fprintf(w, "# TYPE vaultgemma_models_loaded_total counter\n")
			fmt.Fprintf(w, "vaultgemma_models_loaded_total %d\n", totalLoaded)
		}
	}

	fmt.Fprintf(w, "\n")

	fmt.Fprintf(w, "# HELP vaultgemma_memory_alloc_bytes Memory allocated in bytes\n")
	fmt.Fprintf(w, "# TYPE vaultgemma_memory_alloc_bytes gauge\n")
	fmt.Fprintf(w, "vaultgemma_memory_alloc_bytes %d\n\n", m.Alloc)

	fmt.Fprintf(w, "# HELP vaultgemma_goroutines Number of goroutines\n")
	fmt.Fprintf(w, "# TYPE vaultgemma_goroutines gauge\n")
	fmt.Fprintf(w, "vaultgemma_goroutines %d\n\n", runtime.NumGoroutine())

	fmt.Fprintf(w, "# HELP vaultgemma_gc_runs_total Total number of GC runs\n")
	fmt.Fprintf(w, "# TYPE vaultgemma_gc_runs_total counter\n")
	fmt.Fprintf(w, "vaultgemma_gc_runs_total %d\n", m.NumGC)
}

// NewVaultGemmaServer creates a new VaultGemma server instance
func NewVaultGemmaServer(models map[string]*ai.VaultGemma, ggufModels map[string]*gguf.Model, transformerClients map[string]*transformers.Client, domainManager *domain.DomainManager, limiter *rate.Limiter, version string) *VaultGemmaServer {
	inferenceEngine := inference.NewInferenceEngine(models, domainManager)
	enhancedEngine := inference.NewEnhancedInferenceEngine(models, domainManager)

	if transformerClients == nil {
		transformerClients = make(map[string]*transformers.Client)
	}

	// Initialize HANA connection
	hanaPool, err := hanapool.NewPoolFromEnv()
	if err != nil {
		log.Printf("⚠️ Failed to initialize HANA pool: %v", err)
		log.Printf("💡 Continuing without HANA logging and caching")
	}

	var hanaLogger *storage.HANALogger
	var hanaCache hanaCacheStore
	var semanticCache semanticCacheStore
	var enhancedLogging *EnhancedLogging

	if hanaPool != nil {
		// Create HANA logger and cache
		hanaLogger = storage.NewHANALogger(hanaPool)
		hanaCache = storage.NewHANACache(hanaPool)

		// Create semantic cache with custom config
		semanticConfig := &storage.SemanticCacheConfig{
			DefaultTTL:          24 * time.Hour,
			SimilarityThreshold: 0.8,
			MaxEntries:          10000,
			CleanupInterval:     1 * time.Hour,
			EnableVectorSearch:  false,
			EnableFuzzyMatching: true,
		}
		semanticCache = storage.NewSemanticCache(hanaPool, semanticConfig)

		// Create enhanced logging
		enhancedLogging = NewEnhancedLogging(hanaLogger)

		// Create tables if they don't exist
		ctx := context.Background()
		if err := hanaLogger.CreateTables(ctx); err != nil {
			log.Printf("⚠️ Failed to create HANA logger tables: %v", err)
		}
		if err := semanticCache.CreateTables(ctx); err != nil {
			log.Printf("⚠️ Failed to create semantic cache tables: %v", err)
		}
	}

	// Initialize GPU router
	gpuOrchestratorURL := os.Getenv("GPU_ORCHESTRATOR_URL")
	var gpuRouter *gpu.GPURouter
	if gpuOrchestratorURL != "" {
		gpuRouter = gpu.NewGPURouter(gpuOrchestratorURL, nil)
		// Allocate GPUs on startup (estimate 2 GPUs for LocalAI)
		ctx := context.Background()
		requiredGPUs := 2
		if envGPUs := os.Getenv("LOCALAI_GPU_COUNT"); envGPUs != "" {
			// Could parse this, but for now use default
		}
		if err := gpuRouter.AllocateGPUs(ctx, requiredGPUs); err != nil {
			log.Printf("⚠️ Failed to allocate GPUs from orchestrator: %v", err)
			log.Printf("💡 Continuing without GPU orchestration")
		}
	}

	// Initialize enhanced features
	tokenCounter := NewTokenCounter()
	functionRegistry := NewFunctionRegistry()
	retryConfig := DefaultRetryConfig()

	// Initialize performance features
	maxMemoryMB := int64(8192) // Default 8GB
	if envMem := os.Getenv("MODEL_CACHE_MAX_MEMORY_MB"); envMem != "" {
		if parsed, err := strconv.ParseInt(envMem, 10, 64); err == nil && parsed > 0 {
			maxMemoryMB = parsed
		}
	}
	modelCache := NewModelCache(domainManager, maxMemoryMB)
	profiler := NewProfiler(1000)

	// Initialize batch processor (optional, can be enabled via env)
	batchSize := 10
	batchTimeout := 50 * time.Millisecond
	if os.Getenv("ENABLE_REQUEST_BATCHING") == "1" {
		batchProcessor = NewBatchProcessor(batchSize, batchTimeout)
	}

	ocrServices := make(map[string]*vision.DeepSeekOCRService)
	if domainManager != nil {
		configs := domainManager.ListDomainConfigs()
		for name, cfg := range configs {
			if cfg == nil {
				continue
			}
			if !strings.EqualFold(cfg.BackendType, "deepseek-ocr") {
				continue
			}
			visionCfg := cfg.VisionConfig
			if visionCfg == nil {
				log.Printf("⚠️ Vision configuration missing for deepseek-ocr domain %s", name)
				continue
			}
			timeout := time.Duration(visionCfg.TimeoutSeconds) * time.Second
			dsCfg := vision.DeepSeekConfig{
				Domain:        name,
				Endpoint:      firstNonEmpty(visionCfg.Endpoint, cfg.BaseURL),
				APIKey:        firstNonEmpty(visionCfg.APIKey, cfg.APIKey),
				PythonExec:    visionCfg.PythonExec,
				ScriptPath:    visionCfg.ScriptPath,
				ModelVariant:  visionCfg.ModelVariant,
				DefaultPrompt: visionCfg.DefaultPrompt,
				Timeout:       timeout,
			}
			svc, err := vision.NewDeepSeekOCRService(dsCfg)
			if err != nil {
				log.Printf("⚠️ Failed to initialize DeepSeek OCR service for domain %s: %v", name, err)
				continue
			}
			ocrServices[name] = svc
			log.Printf("🔗 DeepSeek OCR service configured for domain %s", name)
		}
	}

	// Initialize Postgres integration (Phase 1)
	var postgresLogger *storage.PostgresInferenceLogger
	var postgresCache *storage.PostgresCacheStore
	postgresDSN := os.Getenv("POSTGRES_DSN")
	if postgresDSN != "" {
		// Initialize inference logger
		if logger, err := storage.NewPostgresInferenceLogger(postgresDSN); err == nil {
			postgresLogger = logger
			log.Printf("✅ Postgres inference logger initialized")
		} else {
			log.Printf("⚠️  Failed to initialize Postgres inference logger: %v", err)
		}

		// Initialize cache store
		if cacheStore, err := storage.NewPostgresCacheStore(postgresDSN); err == nil {
			postgresCache = cacheStore
			log.Printf("✅ Postgres cache store initialized")
			
			// Restore cache state on startup
			ctx := context.Background()
			if states, err := cacheStore.LoadAllCacheStates(ctx); err == nil {
				log.Printf("📦 Restored %d model cache states from Postgres", len(states))
				// Cache states are loaded but not automatically restored to ModelCache
				// This is intentional - models will be loaded on first use
			} else {
				log.Printf("⚠️  Failed to load cache states: %v", err)
			}
		} else {
			log.Printf("⚠️  Failed to initialize Postgres cache store: %v", err)
		}
	}

	return &VaultGemmaServer{
		models:             models,
		ggufModels:         ggufModels,
		transformerClients: transformerClients,
		domainManager:      domainManager,
		limiter:            limiter,
		inferenceEngine:    inferenceEngine,
		enhancedEngine:     enhancedEngine,
		hanaPool:           hanaPool,
		hanaLogger:         hanaLogger,
		hanaCache:          hanaCache,
		semanticCache:      semanticCache,
		enhancedLogging:    enhancedLogging,
		startTime:          time.Now(),
		version:            version,
		// New enhanced features
		tokenCounter:     tokenCounter,
		functionRegistry: functionRegistry,
		retryConfig:      retryConfig,
		ocrServices:      ocrServices,
		gpuRouter:        gpuRouter,
		// Performance features
		modelCache:       modelCache,
		batchProcessor:   batchProcessor,
		profiler:         profiler,
		// Postgres integration (Phase 1)
		postgresLogger:   postgresLogger,
		postgresCache:    postgresCache,
	}
}

// AddModel adds a model for a specific domain
func (s *VaultGemmaServer) AddModel(domainName string, model *ai.VaultGemma) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.models[domainName] = model
}

// DisableEnhancedInference turns off the enhanced inference engine (used in lightweight scenarios/tests).
func (s *VaultGemmaServer) DisableEnhancedInference() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enhancedEngine = nil
}

// GetModels returns all loaded models
func (s *VaultGemmaServer) GetModels() map[string]*ai.VaultGemma {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.models
}

// splitIntoWords splits text into words (simple whitespace-based)
func (s *VaultGemmaServer) splitIntoWords(text string) []string {
	words := make([]string, 0)
	current := strings.Builder{}
	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}
	return words
}

// handleChatWithBatching handles chat requests with batching enabled
func (s *VaultGemmaServer) handleChatWithBatching(w http.ResponseWriter, r *http.Request, ctx context.Context) {
	// Validate and decode request
	req, err := validateChatRequest(r)
	if err != nil {
		handleChatError(w, err, http.StatusBadRequest)
		return
	}

	// Convert to ChatRequest for batch processor
	chatReq := &ChatRequest{
		Model:       req.Model,
		Messages:    convertToChatMessages(req.Messages),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		Domains:     req.Domains,
	}

	// Process through batch processor
	resp, err := s.batchProcessor.ProcessRequest(ctx, chatReq)
	if err != nil {
		handleChatError(w, err, http.StatusInternalServerError)
		return
	}

	// Record batch metrics
	if s.profiler != nil {
		s.profiler.RecordRequest(time.Since(time.Now()))
	}

	// Build and send response
	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(resp)
}

// convertToChatMessages converts internal messages to ChatMessage format
func convertToChatMessages(messages []ChatMessageInternal) []ChatMessage {
	result := make([]ChatMessage, len(messages))
	for i, msg := range messages {
		result[i] = ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}
	return result
}

// hashString generates a simple hash for tokenization
func (s *VaultGemmaServer) hashString(str string) int {
	hash := 0
	for _, r := range str {
		hash = hash*31 + int(r)
	}
	if hash < 0 {
		hash = -hash
	}
	return hash
}
