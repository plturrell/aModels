package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/hanapool"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/gguf"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/transformers"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/vision"
	"golang.org/x/time/rate"
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

func (s *VaultGemmaServer) HandleChat(w http.ResponseWriter, r *http.Request) {
	// Validate HTTP method
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Validate content type
	if r.Header.Get("Content-Type") != "application/json" {
		http.Error(w, "Content-Type must be application/json", http.StatusUnsupportedMediaType)
		return
	}

	// Create context with timeout. Transformers backends can take longer on first use,
	// so allow a generous window.
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	// Track request
	s.mu.Lock()
	s.requestCount++
	s.mu.Unlock()

	start := time.Now()
	var req struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		MaxTokens    int      `json:"max_tokens,omitempty"`
		Temperature  float64  `json:"temperature,omitempty"`
		TopP         float64  `json:"top_p,omitempty"`
		TopK         int      `json:"top_k,omitempty"`
		Domains      []string `json:"domains,omitempty"` // User's available domains
		Images       []string `json:"images,omitempty"`
		VisionPrompt string   `json:"vision_prompt,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("Failed to decode request: %v", err)
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if len(req.Messages) == 0 {
		http.Error(w, "Messages array cannot be empty", http.StatusBadRequest)
		return
	}

	// Check context deadline
	select {
	case <-ctx.Done():
		http.Error(w, "Request timeout", http.StatusRequestTimeout)
		return
	default:
	}

	var prompt string
	for _, msg := range req.Messages {
		prompt += msg.Content + "\n"
	}

	prompt = s.enrichPromptWithAgentCatalog(prompt)

	// Detect or use specified domain
	domain := req.Model
	if domain == "auto" || domain == "" {
		domain = s.domainManager.DetectDomain(prompt, req.Domains)
		log.Printf("Auto-detected domain: %s", domain)
	}

	// Retrieve domain configuration before model resolution
	domainConfig, _ := s.domainManager.GetDomainConfig(domain)
	requireModel := true
	if domainConfig != nil {
		if strings.EqualFold(domainConfig.BackendType, "hf-transformers") {
			requireModel = false
		} else {
			modelPath := strings.ToLower(strings.TrimSpace(domainConfig.ModelPath))
			if strings.HasSuffix(modelPath, ".gguf") {
				requireModel = false
			}
		}
	}

	// Resolve model with fallback and default handling
	modelKey := domain
	model, exists := s.models[modelKey]
	fallbackUsed := false
	fallbackKey := ""
	if !exists && requireModel {
		log.Printf("Domain model not found: %s, attempting fallback", domain)
		if domainConfig != nil && domainConfig.FallbackModel != "" {
			if fallbackModel, ok := s.models[domainConfig.FallbackModel]; ok {
				log.Printf("Using fallback model '%s' for domain '%s'", domainConfig.FallbackModel, domain)
				model = fallbackModel
				modelKey = domainConfig.FallbackModel
				fallbackUsed = true
				fallbackKey = domainConfig.FallbackModel
			}
		}
	}

	if model == nil && requireModel {
		defaultDomain := s.domainManager.GetDefaultDomain()
		if defaultDomain != "" {
			if defaultModel, ok := s.models[defaultDomain]; ok {
				log.Printf("Using default domain '%s' model for request originally targeting '%s'", defaultDomain, domain)
				model = defaultModel
				domain = defaultDomain
				modelKey = defaultDomain
				domainConfig, _ = s.domainManager.GetDomainConfig(defaultDomain)
			}
		}
		if model == nil {
			http.Error(w, fmt.Sprintf("No models available"), http.StatusInternalServerError)
			return
		}
	}

	// Use domain-specific parameters
	maxTokens := req.MaxTokens
	if maxTokens == 0 && domainConfig != nil {
		maxTokens = domainConfig.MaxTokens
	}

	topP, topK := resolveSampling(req.TopP, req.TopK, domainConfig)
	req.TopP = topP
	req.TopK = topK

	metadata := map[string]interface{}{
		"model_key": modelKey,
		"cache_hit": false,
	}
	metadata["top_p"] = topP
	metadata["top_k"] = topK

	// Generate request ID for tracking
	requestID := fmt.Sprintf("req_%d", time.Now().UnixNano())

	// Extract user and session info from headers
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		userID = "anonymous"
	}
	sessionID := r.Header.Get("X-Session-ID")
	if sessionID == "" {
		sessionID = "default"
	}

	// Log request start
	if s.enhancedLogging != nil {
		s.enhancedLogging.LogRequestStart(ctx, requestID, modelKey, domain, prompt, userID, sessionID)
	}

	var content string
	var tokensUsed int
	var cacheHit bool
	var semanticHit bool

	handledExternally := false

	if domainConfig != nil && strings.EqualFold(domainConfig.BackendType, "deepseek-ocr") {
		service := s.ocrServices[domain]
		if service == nil {
			http.Error(w, fmt.Sprintf("OCR service not configured for domain: %s", domain), http.StatusBadGateway)
			return
		}
		if len(req.Images) == 0 {
			http.Error(w, "images array must contain at least one base64 encoded image for deepseek-ocr domains", http.StatusBadRequest)
			return
		}
		imageData, err := decodeImagePayload(req.Images[0])
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to decode image payload: %v", err), http.StatusBadRequest)
			return
		}
		ocrPrompt := req.VisionPrompt
		if ocrPrompt == "" && domainConfig.VisionConfig != nil {
			ocrPrompt = domainConfig.VisionConfig.DefaultPrompt
		}

		text, err := service.ExtractText(ctx, imageData, ocrPrompt)
		if err != nil {
			log.Printf("❌ DeepSeek OCR failed: %v", err)
			if s.enhancedLogging != nil {
				s.enhancedLogging.LogError(ctx, requestID, modelKey, domain, prompt, fmt.Sprintf("DeepSeek OCR failed: %v", err), userID, sessionID, metadata)
			}
			http.Error(w, fmt.Sprintf("deepseek-ocr inference failed: %v", err), http.StatusBadGateway)
			return
		}

		content = text
		tokensUsed = len(strings.Fields(text))
		handledExternally = true
		metadata["backend_type"] = "deepseek-ocr"
		if ocrPrompt != "" {
			metadata["ocr_prompt"] = truncateString(ocrPrompt, 160)
		}
	}

	if !handledExternally && domainConfig != nil && strings.EqualFold(domainConfig.BackendType, "hf-transformers") {
		client := s.transformerClients[domain]
		if client == nil {
			http.Error(w, fmt.Sprintf("transformers service not configured for domain: %s", domain), http.StatusBadGateway)
			return
		}
		chatMessages := make([]transformers.ChatMessage, 0, len(req.Messages))
		for _, msg := range req.Messages {
			if msg.Role == "" || msg.Content == "" {
				continue
			}
			chatMessages = append(chatMessages, transformers.ChatMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
		if len(chatMessages) == 0 {
			http.Error(w, "transformers backend requires at least one message", http.StatusBadRequest)
			return
		}

		generated, used, err := client.Generate(ctx, chatMessages, maxTokens, req.Temperature, topP)
		if err != nil {
			log.Printf("❌ Transformers backend failed for domain %s: %v", domain, err)
			http.Error(w, fmt.Sprintf("transformers backend failed: %v", err), http.StatusBadGateway)
			return
		}
		content = generated
		tokensUsed = used
		handledExternally = true
		metadata["backend_type"] = "hf-transformers"
		if domainConfig.TransformersConfig != nil {
			metadata["transformers_model"] = domainConfig.TransformersConfig.ModelName
		}
	}

	// Check cache first if HANA is available
	if !handledExternally && s.hanaCache != nil {
		// Generate cache key
		cacheKey := s.hanaCache.GenerateCacheKey(prompt, modelKey, domain, req.Temperature, maxTokens, topP, topK)

		// Try to get from cache
		cacheEntry, err := s.hanaCache.Get(ctx, cacheKey)
		if err == nil && cacheEntry != nil {
			content = cacheEntry.Response
			tokensUsed = cacheEntry.TokensUsed
			cacheHit = true
			log.Printf("🎯 Cache hit for domain: %s", domain)

			// Log cache hit
			if s.enhancedLogging != nil {
				s.enhancedLogging.LogCacheHit(ctx, requestID, modelKey, domain, prompt, content, tokensUsed, "exact", 1.0, userID, sessionID)
			}
		}
	}

	// If no exact cache hit, try semantic cache
	if !handledExternally && !cacheHit && s.semanticCache != nil {
		// Find semantically similar cached responses
		similarEntries, err := s.semanticCache.FindSemanticSimilar(ctx, prompt, modelKey, domain, 0.8, 5)
		if err == nil && len(similarEntries) > 0 {
			// Use the most similar entry
			bestEntry := similarEntries[0]
			content = bestEntry.Response
			tokensUsed = bestEntry.TokensUsed
			semanticHit = true
			log.Printf("🧠 Semantic cache hit for domain: %s (similarity: %.2f)", domain, bestEntry.SimilarityScore)

			// Log semantic cache hit
			if s.enhancedLogging != nil {
				s.enhancedLogging.LogCacheHit(ctx, requestID, modelKey, domain, prompt, content, tokensUsed, "semantic", bestEntry.SimilarityScore, userID, sessionID)
			}
		}
	}
	// Attempt GGUF-backed generation when available
	if !handledExternally && !cacheHit && !semanticHit {
		if ggufModel, ok := s.ggufModels[modelKey]; ok && ggufModel != nil {
			temperature := req.Temperature
			if temperature <= 0 {
				if domainConfig != nil && domainConfig.Temperature > 0 {
					temperature = float64(domainConfig.Temperature)
				} else {
					temperature = 0.7
				}
			}
			if maxTokens <= 0 {
				maxTokens = 128
			}

			generated, used, err := ggufModel.Generate(prompt, maxTokens, temperature, topP, topK)
			if err != nil {
				log.Printf("⚠️ GGUF inference failed for %s: %v", modelKey, err)
			} else {
				content = generated
				tokensUsed = used
				metadata["backend_type"] = "gguf"
				handledExternally = true
			}
		}
	}

	// Generate response using actual inference if not cached
	if !handledExternally && !cacheHit && !semanticHit {
		// Use enhanced inference engine if available, otherwise fall back to basic engine
		if s.enhancedEngine != nil {
			enhancedReq := &inference.EnhancedInferenceRequest{
				Prompt:      prompt,
				Domain:      domain,
				MaxTokens:   maxTokens,
				Temperature: req.Temperature,
				Model:       model,
				TopP:        topP,
				TopK:        topK,
			}

			response := s.enhancedEngine.GenerateEnhancedResponse(ctx, enhancedReq)
			if response.Error != nil {
				log.Printf("❌ Enhanced inference failed: %v", response.Error)

				// Log error
				if s.enhancedLogging != nil {
					s.enhancedLogging.LogError(ctx, requestID, modelKey, domain, prompt, fmt.Sprintf("Enhanced inference failed: %v", response.Error), userID, sessionID, metadata)
				}

				http.Error(w, fmt.Sprintf("Enhanced inference failed: %v", response.Error), http.StatusInternalServerError)
				return
			}

			content = response.Content
			tokensUsed = response.TokensUsed
		} else {
			// Fall back to basic inference engine
			if s.inferenceEngine == nil {
				s.inferenceEngine = inference.NewInferenceEngine(s.models, s.domainManager)
			}

			inferenceReq := &inference.InferenceRequest{
				Prompt:      prompt,
				Domain:      domain,
				MaxTokens:   maxTokens,
				Temperature: req.Temperature,
				Model:       model,
				TopP:        topP,
				TopK:        topK,
			}

			response := s.inferenceEngine.GenerateResponse(ctx, inferenceReq)
			if response.Error != nil {
				log.Printf("❌ Inference failed: %v", response.Error)

				// Log error
				if s.enhancedLogging != nil {
					s.enhancedLogging.LogError(ctx, requestID, modelKey, domain, prompt, fmt.Sprintf("Basic inference failed: %v", response.Error), userID, sessionID, metadata)
				}

				http.Error(w, fmt.Sprintf("Inference failed: %v", response.Error), http.StatusInternalServerError)
				return
			}

			content = response.Content
			tokensUsed = response.TokensUsed
		}
	}

	duration := time.Since(start)

	log.Printf("Chat request processed in %.2fms for domain: %s", duration.Seconds()*1000, domain)

	// Update metadata with final values
	if _, ok := metadata["backend_type"]; !ok {
		metadata["backend_type"] = "vaultgemma"
	}
	metadata["cache_hit"] = cacheHit
	if domainConfig != nil {
		metadata["domain_name"] = domainConfig.Name
	}
	if fallbackUsed {
		metadata["fallback_used"] = true
		metadata["fallback_model"] = fallbackKey

		// Log model switch
		if s.enhancedLogging != nil {
			s.enhancedLogging.LogModelSwitch(ctx, requestID, modelKey, fallbackKey, domain, prompt, "model_unavailable", userID, sessionID)
		}
	} else {
		metadata["fallback_used"] = false
	}

	// Log request completion with enhanced logging
	if s.enhancedLogging != nil {
		s.enhancedLogging.LogRequestEnd(ctx, requestID, modelKey, domain, prompt, content, tokensUsed, duration.Milliseconds(), req.Temperature, maxTokens, cacheHit, semanticHit, userID, sessionID, metadata)
	}

	// Cache the response if not already cached and HANA is available
	if !handledExternally && !cacheHit && !semanticHit && s.hanaCache != nil {
		cacheKey := s.hanaCache.GenerateCacheKey(prompt, modelKey, domain, req.Temperature, maxTokens, topP, topK)
		cacheEntry := &storage.CacheEntry{
			CacheKey:    cacheKey,
			PromptHash:  fmt.Sprintf("%x", []byte(prompt)),
			Model:       modelKey,
			Domain:      domain,
			Response:    content,
			TokensUsed:  tokensUsed,
			Temperature: req.Temperature,
			MaxTokens:   maxTokens,
		}

		// Cache asynchronously
		go func() {
			if err := s.hanaCache.Set(context.Background(), cacheEntry); err != nil {
				log.Printf("⚠️ Failed to cache response: %v", err)
			}
		}()
	}

	// Also store in semantic cache for future similarity matching
	if !handledExternally && !cacheHit && !semanticHit && s.semanticCache != nil {
		semanticCacheKey := s.semanticCache.GenerateCacheKey(prompt, modelKey, domain, req.Temperature, maxTokens, topP, topK)
		semanticEntry := &storage.SemanticCacheEntry{
			CacheKey:        semanticCacheKey,
			PromptHash:      fmt.Sprintf("%x", []byte(prompt)),
			SemanticHash:    s.semanticCache.GenerateSemanticHash(prompt),
			Model:           modelKey,
			Domain:          domain,
			Prompt:          prompt,
			Response:        content,
			TokensUsed:      tokensUsed,
			Temperature:     req.Temperature,
			MaxTokens:       maxTokens,
			SimilarityScore: 1.0,
			Metadata: map[string]string{
				"user_id":    "anonymous",
				"session_id": "default",
				"source":     "inference",
			},
			Tags: []string{domain, "inference", "generated"},
		}

		// Cache asynchronously
		go func() {
			if err := s.semanticCache.Set(context.Background(), semanticEntry); err != nil {
				log.Printf("⚠️ Failed to cache semantic response: %v", err)
			}
		}()
	}

	resp := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   modelKey,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": content,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     len(prompt) / 4,
			"completion_tokens": tokensUsed,
			"total_tokens":      (len(prompt) / 4) + tokensUsed,
		},
		"metadata": metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
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

	w.Header().Set("Content-Type", "application/json")
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

	w.Header().Set("Content-Type", "application/json")
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

	w.Header().Set("Content-Type", "application/json")
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

	for domain, model := range s.models {
		config, _ := s.domainManager.GetDomainConfig(domain)

		domainInfo := map[string]interface{}{
			"id":          domain,
			"loaded":      true,
			"layers":      model.Config.NumLayers,
			"hidden_size": model.Config.HiddenSize,
		}

		if config != nil {
			domainInfo["name"] = config.Name
			domainInfo["max_tokens"] = config.MaxTokens
			domainInfo["tags"] = config.DomainTags
			domainInfo["fallback_model"] = config.FallbackModel
		}

		domains = append(domains, domainInfo)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   domains,
	})
}

// handleMetrics provides Prometheus-style metrics endpoint
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
	fmt.Fprintf(w, "vaultgemma_uptime_seconds %.2f\n\n", uptime.Seconds())

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

	// Initialize enhanced features
	tokenCounter := NewTokenCounter()
	functionRegistry := NewFunctionRegistry()
	retryConfig := DefaultRetryConfig()

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
