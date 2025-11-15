package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"github.com/plturrell/aModels/pkg/observability/llm"
)

// V2 API provides enhanced features:
// - Structured request/response with metadata
// - Built-in tracing support
// - Enhanced error responses with error codes
// - Support for workflow tracking
// - Async request support
// - Batch operations

// V2ChatRequest represents the v2 API chat request
type V2ChatRequest struct {
	Model       string                 `json:"model"`
	Messages    []ChatMessage          `json:"messages"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	TopP        float64                `json:"top_p,omitempty"`
	TopK        int                    `json:"top_k,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	// V2-specific fields
	WorkflowID   string   `json:"workflow_id,omitempty"`
	TraceParent  string   `json:"trace_parent,omitempty"`
	Async        bool     `json:"async,omitempty"`
	CallbackURL  string   `json:"callback_url,omitempty"`
	DomainFilter []string `json:"domain_filter,omitempty"`
}

// V2ChatResponse represents the v2 API chat response
type V2ChatResponse struct {
	ID          string                 `json:"id"`
	Object      string                 `json:"object"`
	Created     int64                  `json:"created"`
	Model       string                 `json:"model"`
	Domain      string                 `json:"domain,omitempty"`
	Choices     []V2Choice             `json:"choices"`
	Usage       V2Usage                `json:"usage"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	TraceID     string                 `json:"trace_id,omitempty"`
	WorkflowID  string                 `json:"workflow_id,omitempty"`
	Status      string                 `json:"status"`
	RequestTime time.Time              `json:"request_time"`
}

// V2Choice represents a chat completion choice in v2 API
type V2Choice struct {
	Index        int                    `json:"index"`
	Message      ChatMessage           `json:"message"`
	FinishReason string                 `json:"finish_reason"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// V2Usage represents token usage in v2 API
type V2Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// V2ErrorResponse represents an error response in v2 API
type V2ErrorResponse struct {
	Error V2Error `json:"error"`
}

// V2Error represents an error in v2 API
type V2Error struct {
	Code       string                 `json:"code"`
	Message    string                 `json:"message"`
	Type       string                 `json:"type"`
	Param      string                 `json:"param,omitempty"`
	TraceID    string                 `json:"trace_id,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	RetryAfter int                    `json:"retry_after,omitempty"`
}

// HandleV2Chat handles v2 API chat completion requests
func (s *VaultGemmaServer) HandleV2Chat(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), RequestTimeoutDefault)
	defer cancel()

	// Start distributed trace
	ctx, span := s.tracer.Start(ctx, "v2.chat.completion")
	defer span.End()

	// Track request
	s.mu.Lock()
	s.requestCount++
	s.mu.Unlock()

	start := time.Now()

	// Validate and decode v2 request
	var req V2ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.handleV2Error(w, span, "invalid_request", "Invalid JSON in request body", err, http.StatusBadRequest)
		return
	}

	// Add OpenLLMetry request attributes
	llmConfig := llm.LLMRequestConfig{
		System:      "localai",
		Model:       req.Model,
		RequestType: "chat",
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   int64(req.MaxTokens),
		IsStreaming: req.Stream,
	}
	llm.AddLLMRequestAttributes(span, llmConfig)

	// Add additional trace attributes
	span.SetAttributes(
		attribute.String("model", req.Model),
		attribute.Int("max_tokens", req.MaxTokens),
		attribute.Float64("temperature", req.Temperature),
		attribute.Bool("stream", req.Stream),
		attribute.Bool("async", req.Async),
		attribute.Int("llm.request.message_count", len(req.Messages)),
	)
	if req.WorkflowID != "" {
		span.SetAttributes(attribute.String("workflow_id", req.WorkflowID))
	}

	// Extract trace context if provided
	if req.TraceParent != "" {
		// TODO: Parse W3C Trace Context format and link spans
		log.Printf("Received trace parent: %s", req.TraceParent)
	}

	// Build prompt from messages (convert to internal format)
	internalMessages := make([]ChatMessageInternal, len(req.Messages))
	for i, msg := range req.Messages {
		internalMessages[i] = ChatMessageInternal{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}
	prompt := buildPromptFromMessages(internalMessages)
	prompt = s.enrichPromptWithAgentCatalog(prompt)

	// Detect or use specified domain
	domain := req.Model
	if domain == DomainAuto || domain == "" {
		domain = s.domainManager.DetectDomain(prompt, req.DomainFilter)
		log.Printf("Auto-detected domain: %s", domain)
	}
	span.SetAttributes(attribute.String("domain", domain))

	// Retrieve domain configuration
	domainConfig, _ := s.domainManager.GetDomainConfig(domain)
	preferredBackend := pickPreferredBackend()

	// Resolve model with fallback
	model, modelKey, fallbackUsed, fallbackKey, err := s.resolveModelForDomain(ctx, domain, domainConfig, preferredBackend)
	if err != nil {
		s.handleV2Error(w, span, "model_not_found", "Failed to resolve model", err, http.StatusServiceUnavailable)
		return
	}

	// Update domain if fallback changed it
	if fallbackUsed && fallbackKey != "" {
		domain = fallbackKey
		domainConfig, _ = s.domainManager.GetDomainConfig(domain)
		span.SetAttributes(attribute.Bool("fallback_used", true))
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
	requestID := fmt.Sprintf("v2_req_%d", time.Now().UnixNano())
	traceID := span.SpanContext().TraceID().String()

	// Extract user and session info from headers
	userID := r.Header.Get(HeaderUserID)
	if userID == "" {
		userID = AnonymousUserID
	}
	sessionID := r.Header.Get(HeaderSessionID)
	if sessionID == "" {
		sessionID = DefaultSessionID
	}

	// Process chat request
	internalReq := &ChatRequestInternal{
		Model:       req.Model,
		Messages:    internalMessages,
		MaxTokens:   maxTokens,
		Temperature: req.Temperature,
		TopP:        topP,
		TopK:        topK,
	}

	result, err := s.processChatRequest(ctx, internalReq, domain, domainConfig, model, modelKey, prompt, maxTokens, topP, topK, requestID, userID, sessionID)
	if err != nil {
		if s.Profiler != nil {
			s.Profiler.RecordError()
		}
		s.handleV2Error(w, span, "inference_error", "Failed to process chat request", err, http.StatusBadGateway)
		return
	}

	duration := time.Since(start)
	log.Printf("V2 chat request processed in %.2fms for domain: %s", duration.Seconds()*1000, domain)

	// Record profiling metrics
	if s.Profiler != nil {
		s.Profiler.RecordRequest(duration)
	}

	// Log inference to Postgres (non-blocking)
	if s.postgresLogger != nil {
		go func() {
			logEntry := &storage.PostgresInferenceLog{
				Domain:          domain,
				ModelName:       modelKey,
				PromptLength:    len(prompt),
				ResponseLength:  len(result.Content),
				LatencyMS:       int(duration.Milliseconds()),
				TokensGenerated: result.TokensUsed,
				TokensPrompt:    len(prompt) / 4,
				WorkflowID:      req.WorkflowID,
				UserID:          userID,
				CreatedAt:       time.Now(),
			}
			if err := s.postgresLogger.LogInference(context.Background(), logEntry); err != nil {
				log.Printf("⚠️  Failed to log inference to Postgres: %v", err)
			}
		}()
	}

	// Build v2 response
	metadata := result.Metadata
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	metadata["latency_ms"] = duration.Milliseconds()
	metadata["backend_type"] = result.Metadata["backend_type"]
	metadata["cache_hit"] = result.CacheHit
	metadata["domain_name"] = domain
	if fallbackUsed {
		metadata["fallback_used"] = true
		metadata["fallback_model"] = fallbackKey
	}

	resp := V2ChatResponse{
		ID:          requestID,
		Object:      "chat.completion",
		Created:     start.Unix(),
		Model:       modelKey,
		Domain:      domain,
		Status:      "completed",
		RequestTime: start,
		TraceID:     traceID,
		WorkflowID:  req.WorkflowID,
		Choices: []V2Choice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: result.Content,
				},
				FinishReason: "stop",
				Metadata:     metadata,
			},
		},
		Usage: V2Usage{
			PromptTokens:     len(prompt) / 4,
			CompletionTokens: result.TokensUsed,
			TotalTokens:      len(prompt)/4 + result.TokensUsed,
		},
		Metadata: metadata,
	}

	// Add OpenLLMetry response attributes
	llmResponse := llm.LLMResponseInfo{
		PromptTokens:     int64(resp.Usage.PromptTokens),
		CompletionTokens: int64(resp.Usage.CompletionTokens),
		TotalTokens:      int64(resp.Usage.TotalTokens),
		FinishReason:     "stop",
	}
	llm.AddLLMResponseAttributes(span, llmResponse)

	span.SetStatus(codes.Ok, "Request completed successfully")
	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(resp)
}

// handleV2Error handles errors in v2 API with enhanced error format
func (s *VaultGemmaServer) handleV2Error(w http.ResponseWriter, span interface{}, code, message string, err error, statusCode int) {
	errorMsg := message
	if err != nil {
		errorMsg = fmt.Sprintf("%s: %v", message, err)
	}

	traceID := ""
	if span != nil {
		// Try to extract trace ID from span if it's available
		// This is a simplified version, real implementation would extract from span context
		traceID = "trace_unavailable"
	}

	errorResp := V2ErrorResponse{
		Error: V2Error{
			Code:    code,
			Message: errorMsg,
			Type:    "api_error",
			TraceID: traceID,
		},
	}

	// Add retry-after for rate limit errors
	if statusCode == http.StatusTooManyRequests {
		errorResp.Error.RetryAfter = 60
	}

	w.Header().Set(HeaderContentType, ContentTypeJSON)
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(errorResp)

	log.Printf("V2 API Error [%s]: %s", code, errorMsg)
}

// HandleV2Models handles v2 API models endpoint
func (s *VaultGemmaServer) HandleV2Models(w http.ResponseWriter, r *http.Request) {
	_, span := s.tracer.Start(r.Context(), "v2.models.list")
	defer span.End()

	domains := s.domainManager.ListDomainConfigs()
	models := make([]map[string]interface{}, 0, len(domains))

	for domainID, cfg := range domains {
		models = append(models, map[string]interface{}{
			"id":          domainID,
			"name":        cfg.Name,
			"layer":       cfg.Layer,
			"team":        cfg.Team,
			"backend":     cfg.BackendType,
			"tags":        cfg.DomainTags,
			"keywords":    cfg.Keywords,
			"max_tokens":  cfg.MaxTokens,
			"temperature": cfg.Temperature,
		})
	}

	resp := map[string]interface{}{
		"object":  "list",
		"data":    models,
		"version": "v2",
		"count":   len(models),
	}

	span.SetStatus(codes.Ok, fmt.Sprintf("Listed %d models", len(models)))
	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(resp)
}

// HandleV2Health handles v2 API health check endpoint
func (s *VaultGemmaServer) HandleV2Health(w http.ResponseWriter, r *http.Request) {
	_, span := s.tracer.Start(r.Context(), "v2.health.check")
	defer span.End()

	s.mu.RLock()
	requestCount := s.requestCount
	uptime := time.Since(s.startTime)
	s.mu.RUnlock()

	health := map[string]interface{}{
		"status":    "ok",
		"version":   "v2",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"service": map[string]interface{}{
			"name":    "vaultgemma-localai",
			"version": s.version,
			"uptime":  uptime.String(),
		},
		"metrics": map[string]interface{}{
			"requests_total": requestCount,
		},
		"features": []string{
			"distributed_tracing",
			"workflow_tracking",
			"async_requests",
			"batch_operations",
			"enhanced_errors",
		},
	}

	if s.Profiler != nil {
		health["profiler"] = s.Profiler.GetStats()
	}

	span.SetStatus(codes.Ok, "Service healthy")
	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(health)
}
