// Package server chat helpers provides helper functions for processing chat completion requests.
// These functions handle request validation, model resolution, backend processing,
// caching, and response building. The helpers are used by the main HandleChat function
// to keep the code modular and maintainable.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/gguf"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/transformers"
)

// ChatRequestInternal represents the internal chat request structure
type ChatRequestInternal struct {
	Model       string
	Messages    []ChatMessageInternal
	MaxTokens   int
	Temperature float64
	TopP        float64
	TopK        int
	Domains     []string
	Images      []string
	VisionPrompt string
}

// ChatMessageInternal represents a message in the internal request
type ChatMessageInternal struct {
	Role    string
	Content string
}

// ChatProcessingResult contains the result of processing a chat request
type ChatProcessingResult struct {
	Content      string
	TokensUsed   int
	CacheHit     bool
	SemanticHit  bool
	HandledExternally bool
	BackendType  string
	ModelKey     string
	Domain       string
	DomainConfig *domain.DomainConfig
	Metadata     map[string]interface{}
	Error        error
}

// validateChatRequest validates the HTTP request and decodes the JSON body
func validateChatRequest(r *http.Request) (*ChatRequestInternal, error) {
	// Validate HTTP method
	if r.Method != http.MethodPost {
		return nil, fmt.Errorf("%w: method %s not allowed, expected POST", ErrInvalidRequest, r.Method)
	}

	// Validate content type
	if r.Header.Get(HeaderContentType) != ContentTypeJSON {
		return nil, fmt.Errorf("%w: Content-Type must be %s", ErrInvalidRequest, ContentTypeJSON)
	}

	// Decode request body
	var req struct {
		Model       string `json:"model"`
		Messages    []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		MaxTokens    int      `json:"max_tokens,omitempty"`
		Temperature  float64  `json:"temperature,omitempty"`
		TopP         float64  `json:"top_p,omitempty"`
		TopK         int      `json:"top_k,omitempty"`
		Domains      []string `json:"domains,omitempty"`
		Images       []string `json:"images,omitempty"`
		VisionPrompt string   `json:"vision_prompt,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return nil, fmt.Errorf("%w: failed to decode JSON body: %w", ErrInvalidRequest, err)
	}

	// Validate request
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("%w: messages array cannot be empty", ErrInvalidRequest)
	}

	// Convert to internal format
	messages := make([]ChatMessageInternal, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = ChatMessageInternal{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	return &ChatRequestInternal{
		Model:       req.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		Domains:     req.Domains,
		Images:      req.Images,
		VisionPrompt: req.VisionPrompt,
	}, nil
}

// buildPromptFromMessages builds a prompt string from chat messages
func buildPromptFromMessages(messages []ChatMessageInternal) string {
	var prompt strings.Builder
	for _, msg := range messages {
		prompt.WriteString(msg.Content)
		prompt.WriteString("\n")
	}
	return prompt.String()
}

// resolveModelForDomain resolves the model for a given domain with fallback logic
func (s *VaultGemmaServer) resolveModelForDomain(
	ctx context.Context,
	domain string,
	domainConfig *domain.DomainConfig,
	preferredBackend string,
) (model *ai.VaultGemma, modelKey string, fallbackUsed bool, fallbackKey string, err error) {
	// Determine if we need a safetensors model
	requireModel := true
	if domainConfig != nil {
		if strings.EqualFold(domainConfig.BackendType, BackendTypeTransformers) {
			requireModel = false
		} else {
			modelPath := strings.ToLower(strings.TrimSpace(domainConfig.ModelPath))
			if strings.HasSuffix(modelPath, ".gguf") {
				requireModel = false
			}
		}
	}

	if !requireModel {
		return nil, domain, false, "", nil
	}

	// Phase 4: Try lazy loading from cache first
	if s.ModelCache != nil {
		// Check if this is a GGUF model
		if domainConfig != nil {
			modelPath := strings.ToLower(strings.TrimSpace(domainConfig.ModelPath))
			if strings.HasSuffix(modelPath, ".gguf") {
				// GGUF models are handled separately in processChatRequest
				return nil, domain, false, "", nil
			}
		}

		// Try to get safetensors model from cache (lazy loading)
		cachedModel, err := s.ModelCache.GetSafetensorModel(ctx, domain)
		if err == nil && cachedModel != nil {
			return cachedModel, domain, false, "", nil
		}

		// Try fallback model from cache
		if domainConfig != nil && domainConfig.FallbackModel != "" {
			fallbackModel, err := s.ModelCache.GetSafetensorModel(ctx, domainConfig.FallbackModel)
			if err == nil && fallbackModel != nil {
				log.Printf("Using fallback model '%s' for domain '%s' (lazy loaded)", domainConfig.FallbackModel, domain)
				return fallbackModel, domainConfig.FallbackModel, true, domainConfig.FallbackModel, nil
			}
		}

		// Try default domain from cache
		defaultDomain := s.domainManager.GetDefaultDomain()
		if defaultDomain != "" {
			defaultModel, err := s.ModelCache.GetSafetensorModel(ctx, defaultDomain)
			if err == nil && defaultModel != nil {
				log.Printf("Using default domain '%s' model for request originally targeting '%s' (lazy loaded)", defaultDomain, domain)
				return defaultModel, defaultDomain, false, "", nil
			}
		}
	}

	// Fallback to pre-loaded models (for backward compatibility)
	modelKey = domain
	model, exists := s.models[modelKey]

	// Try fallback if model not found
	if !exists {
		log.Printf("Domain model not found: %s, attempting fallback", domain)
		if domainConfig != nil && domainConfig.FallbackModel != "" {
			if fallbackModel, ok := s.models[domainConfig.FallbackModel]; ok {
				log.Printf("Using fallback model '%s' for domain '%s'", domainConfig.FallbackModel, domain)
				return fallbackModel, domainConfig.FallbackModel, true, domainConfig.FallbackModel, nil
			}
		}
	}

	// Try default domain if still no model
	if model == nil {
		defaultDomain := s.domainManager.GetDefaultDomain()
		if defaultDomain != "" {
			if defaultModel, ok := s.models[defaultDomain]; ok {
				log.Printf("Using default domain '%s' model for request originally targeting '%s'", defaultDomain, domain)
				return defaultModel, defaultDomain, false, "", nil
			}
		}
		return nil, "", false, "", fmt.Errorf("%w: no models available for domain %s", ErrModelNotFound, domain)
	}

	return model, modelKey, false, "", nil
}

// processChatRequest processes the chat request and generates a response
func (s *VaultGemmaServer) processChatRequest(
	ctx context.Context,
	req *ChatRequestInternal,
	domain string,
	domainConfig *domain.DomainConfig,
	model *ai.VaultGemma,
	modelKey string,
	prompt string,
	maxTokens int,
	topP float64,
	topK int,
	requestID string,
	userID string,
	sessionID string,
) (*ChatProcessingResult, error) {
	result := &ChatProcessingResult{
		ModelKey:    modelKey,
		Domain:      domain,
		DomainConfig: domainConfig,
		Metadata: map[string]interface{}{
			"model_key": modelKey,
			"cache_hit": false,
			"top_p":     topP,
			"top_k":     topK,
		},
	}

	// Determine backend type
	backendType := ""
	if domainConfig != nil {
		backendType = strings.TrimSpace(domainConfig.BackendType)
	}
	if backendType == "" {
		backendType = pickPreferredBackend()
	}
	result.BackendType = backendType

	// Handle DeepSeek OCR backend
	if strings.EqualFold(backendType, BackendTypeDeepSeekOCR) {
		return s.processDeepSeekOCR(ctx, req, domain, domainConfig, prompt, requestID, userID, sessionID, result)
	}

	// Handle Transformers backend
	if strings.EqualFold(backendType, BackendTypeTransformers) {
		return s.processTransformersBackend(ctx, req, domain, domainConfig, maxTokens, topP, prompt, requestID, userID, sessionID, result)
	}

	// Check cache first if Postgres cache is available
	if s.postgresCache != nil {
		cacheResult, err := s.checkCache(ctx, prompt, modelKey, domain, req.Temperature, maxTokens, topP, topK, requestID, userID, sessionID)
		if err == nil && cacheResult != nil {
			result.Content = cacheResult.Content
			result.TokensUsed = cacheResult.TokensUsed
			result.CacheHit = cacheResult.CacheHit
			result.SemanticHit = cacheResult.SemanticHit
			return result, nil
		}
	}

	// Try GGUF backend (with lazy loading)
	var ggufModel *gguf.Model
	if s.ggufModels[modelKey] != nil {
		ggufModel = s.ggufModels[modelKey]
	} else if s.ModelCache != nil {
		// Phase 4: Try lazy loading GGUF model from cache
		loadedGGUF, err := s.ModelCache.GetGGUFModel(ctx, modelKey)
		if err == nil && loadedGGUF != nil {
			ggufModel = loadedGGUF
			// Cache it for future use
			s.ggufModels[modelKey] = loadedGGUF
		}
	}

	if ggufModel != nil {
		ggufResult, err := s.processGGUFBackend(ctx, ggufModel, prompt, maxTokens, req.Temperature, topP, topK, modelKey)
		if err == nil {
			result.Content = ggufResult.Content
			result.TokensUsed = ggufResult.TokensUsed
			result.HandledExternally = true
			result.BackendType = BackendTypeGGUF
			return result, nil
		}
	}

	// Generate response using inference engine
	return s.processInferenceBackend(ctx, model, prompt, domain, maxTokens, req.Temperature, topP, topK, requestID, userID, sessionID, result)
}

// processDeepSeekOCR handles DeepSeek OCR backend processing
func (s *VaultGemmaServer) processDeepSeekOCR(
	ctx context.Context,
	req *ChatRequestInternal,
	domain string,
	domainConfig *domain.DomainConfig,
	prompt string,
	requestID string,
	userID string,
	sessionID string,
	result *ChatProcessingResult,
) (*ChatProcessingResult, error) {
	service := s.ocrServices[domain]
	if service == nil {
		return nil, fmt.Errorf("%w: OCR service not configured for domain: %s", ErrBackendUnavailable, domain)
	}

	if len(req.Images) == 0 {
		return nil, fmt.Errorf("%w: images array must contain at least one base64 encoded image for deepseek-ocr domains", ErrInvalidRequest)
	}

	imageData, err := decodeImagePayload(req.Images[0])
	if err != nil {
		return nil, fmt.Errorf("%w: failed to decode image payload: %w", ErrInvalidRequest, err)
	}

	ocrPrompt := req.VisionPrompt
	if ocrPrompt == "" && domainConfig.VisionConfig != nil {
		ocrPrompt = domainConfig.VisionConfig.DefaultPrompt
	}

	text, err := service.ExtractText(ctx, imageData, ocrPrompt)
	if err != nil {
		log.Printf("❌ DeepSeek OCR failed: %v", err)
		if s.enhancedLogging != nil {
			s.enhancedLogging.LogError(ctx, requestID, result.ModelKey, domain, prompt, fmt.Sprintf("DeepSeek OCR failed: %v", err), userID, sessionID, result.Metadata)
		}
		return nil, fmt.Errorf("%w: deepseek-ocr inference failed: %w", ErrBackendUnavailable, err)
	}

	result.Content = text
	result.TokensUsed = len(strings.Fields(text))
	result.HandledExternally = true
	result.BackendType = BackendTypeDeepSeekOCR
	result.Metadata["backend_type"] = BackendTypeDeepSeekOCR
	if ocrPrompt != "" {
		result.Metadata["ocr_prompt"] = truncateString(ocrPrompt, 160)
	}

	return result, nil
}

// processTransformersBackend handles HuggingFace transformers backend processing
func (s *VaultGemmaServer) processTransformersBackend(
	ctx context.Context,
	req *ChatRequestInternal,
	domain string,
	domainConfig *domain.DomainConfig,
	maxTokens int,
	topP float64,
	prompt string,
	requestID string,
	userID string,
	sessionID string,
	result *ChatProcessingResult,
) (*ChatProcessingResult, error) {
	client := s.transformerClients[domain]
	if client == nil {
		return nil, fmt.Errorf("%w: transformers service not configured for domain: %s", ErrBackendUnavailable, domain)
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
		return nil, fmt.Errorf("%w: transformers backend requires at least one message", ErrInvalidRequest)
	}

	generated, used, err := client.Generate(ctx, chatMessages, maxTokens, req.Temperature, topP)
	if err != nil {
		log.Printf("❌ Transformers backend failed for domain %s: %v", domain, err)
		return nil, fmt.Errorf("%w: transformers backend failed: %w", ErrBackendUnavailable, err)
	}

	result.Content = generated
	result.TokensUsed = used
	result.HandledExternally = true
	result.BackendType = BackendTypeTransformers
	result.Metadata["backend_type"] = BackendTypeTransformers
	if domainConfig.TransformersConfig != nil {
		result.Metadata["transformers_model"] = domainConfig.TransformersConfig.ModelName
	}

	return result, nil
}

// cacheResult represents a cache lookup result
type cacheResult struct {
	Content     string
	TokensUsed  int
	CacheHit    bool
	SemanticHit bool
}

// checkCache checks both HANA cache and semantic cache
func (s *VaultGemmaServer) checkCache(
	ctx context.Context,
	prompt string,
	modelKey string,
	domain string,
	temperature float64,
	maxTokens int,
	topP float64,
	topK int,
	requestID string,
	userID string,
	sessionID string,
) (*cacheResult, error) {
	// Try exact cache first
	if s.postgresCache != nil {
		cacheKey := s.postgresCache.GenerateCacheKey(prompt, modelKey, domain, temperature, maxTokens, topP, topK)
		cacheEntry, err := s.postgresCache.Get(ctx, cacheKey)
		if err == nil && cacheEntry != nil {
			log.Printf("🎯 Cache hit for domain: %s", domain)
			if s.enhancedLogging != nil {
				s.enhancedLogging.LogCacheHit(ctx, requestID, modelKey, domain, prompt, cacheEntry.Response, cacheEntry.TokensUsed, "exact", 1.0, userID, sessionID)
			}
			return &cacheResult{
				Content:    cacheEntry.Response,
				TokensUsed: cacheEntry.TokensUsed,
				CacheHit:   true,
			}, nil
		}
	}

	// Try semantic cache
	if s.semanticCache != nil {
		similarEntries, err := s.semanticCache.FindSemanticSimilar(ctx, prompt, modelKey, domain, 0.8, 5)
		if err == nil && len(similarEntries) > 0 {
			bestEntry := similarEntries[0]
			log.Printf("🧠 Semantic cache hit for domain: %s (similarity: %.2f)", domain, bestEntry.SimilarityScore)
			if s.enhancedLogging != nil {
				s.enhancedLogging.LogCacheHit(ctx, requestID, modelKey, domain, prompt, bestEntry.Response, bestEntry.TokensUsed, "semantic", bestEntry.SimilarityScore, userID, sessionID)
			}
			return &cacheResult{
				Content:     bestEntry.Response,
				TokensUsed:  bestEntry.TokensUsed,
				SemanticHit: true,
			}, nil
		}
	}

	return nil, fmt.Errorf("no cache hit")
}

// ggufResult represents a GGUF generation result
type ggufResult struct {
	Content    string
	TokensUsed int
}

// processGGUFBackend handles GGUF backend processing
func (s *VaultGemmaServer) processGGUFBackend(
	ctx context.Context,
	ggufModel *gguf.Model,
	prompt string,
	maxTokens int,
	temperature float64,
	topP float64,
	topK int,
	modelKey string,
) (*ggufResult, error) {
	if temperature <= 0 {
		temperature = DefaultTemperature
	}
	if maxTokens <= 0 {
		maxTokens = 128
	}

	generated, used, err := ggufModel.Generate(prompt, maxTokens, temperature, topP, topK)
	if err != nil {
		log.Printf("⚠️ GGUF inference failed for %s: %v", modelKey, err)
		return nil, err
	}

	return &ggufResult{
		Content:    generated,
		TokensUsed: used,
	}, nil
}

// processInferenceBackend handles safetensors inference backend processing
func (s *VaultGemmaServer) processInferenceBackend(
	ctx context.Context,
	model *ai.VaultGemma,
	prompt string,
	domain string,
	maxTokens int,
	temperature float64,
	topP float64,
	topK int,
	requestID string,
	userID string,
	sessionID string,
	result *ChatProcessingResult,
) (*ChatProcessingResult, error) {
	// Use enhanced inference engine if available
	if s.enhancedEngine != nil {
		enhancedReq := &inference.EnhancedInferenceRequest{
			Prompt:      prompt,
			Domain:      domain,
			MaxTokens:   maxTokens,
			Temperature: temperature,
			Model:       model,
			TopP:        topP,
			TopK:        topK,
		}

		response := s.enhancedEngine.GenerateEnhancedResponse(ctx, enhancedReq)
		if response.Error != nil {
			log.Printf("❌ Enhanced inference failed: %v", response.Error)
			if s.enhancedLogging != nil {
				s.enhancedLogging.LogError(ctx, requestID, result.ModelKey, domain, prompt, fmt.Sprintf("Enhanced inference failed: %v", response.Error), userID, sessionID, result.Metadata)
			}
			return nil, fmt.Errorf("%w: enhanced inference failed: %w", ErrInternalError, response.Error)
		}

		result.Content = response.Content
		result.TokensUsed = response.TokensUsed
		result.BackendType = BackendTypeSafetensors
		return result, nil
	}

	// Fall back to basic inference engine
	if s.inferenceEngine == nil {
		s.inferenceEngine = inference.NewInferenceEngine(s.models, s.domainManager)
	}

	inferenceReq := &inference.InferenceRequest{
		Prompt:      prompt,
		Domain:      domain,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Model:       model,
		TopP:        topP,
		TopK:        topK,
	}

	response := s.inferenceEngine.GenerateResponse(ctx, inferenceReq)
	if response.Error != nil {
		log.Printf("❌ Inference failed: %v", response.Error)
		if s.enhancedLogging != nil {
			s.enhancedLogging.LogError(ctx, requestID, result.ModelKey, domain, prompt, fmt.Sprintf("Basic inference failed: %v", response.Error), userID, sessionID, result.Metadata)
		}
		return nil, fmt.Errorf("%w: inference failed: %w", ErrInternalError, response.Error)
	}

	result.Content = response.Content
	result.TokensUsed = response.TokensUsed
	result.BackendType = BackendTypeSafetensors
	return result, nil
}

// buildChatResponse builds the final HTTP response for a chat completion
func buildChatResponse(
	modelKey string,
	content string,
	tokensUsed int,
	prompt string,
	metadata map[string]interface{},
) map[string]interface{} {
	return map[string]interface{}{
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
}

// saveToCache saves the response to both HANA and semantic caches
func (s *VaultGemmaServer) saveToCache(
	ctx context.Context,
	prompt string,
	modelKey string,
	domain string,
	content string,
	tokensUsed int,
	temperature float64,
	maxTokens int,
	topP float64,
	topK int,
) {
	// Save to Postgres cache
	if s.postgresCache != nil {
		cacheKey := s.postgresCache.GenerateCacheKey(prompt, modelKey, domain, temperature, maxTokens, topP, topK)
		cacheEntry := &storage.CacheEntry{
			CacheKey:    cacheKey,
			PromptHash:  fmt.Sprintf("%x", []byte(prompt)),
			Model:       modelKey,
			Domain:      domain,
			Response:    content,
			TokensUsed:  tokensUsed,
			Temperature: temperature,
			MaxTokens:   maxTokens,
		}

		go func() {
			if err := s.postgresCache.Set(context.Background(), cacheEntry); err != nil {
				log.Printf("⚠️ Failed to cache response: %v", err)
			}
		}()
	}

	// Save to semantic cache
	if s.semanticCache != nil {
		semanticCacheKey := s.semanticCache.GenerateCacheKey(prompt, modelKey, domain, temperature, maxTokens, topP, topK)
		semanticEntry := &storage.SemanticCacheEntry{
			CacheKey:        semanticCacheKey,
			PromptHash:      fmt.Sprintf("%x", []byte(prompt)),
			SemanticHash:    s.semanticCache.GenerateSemanticHash(prompt),
			Model:           modelKey,
			Domain:          domain,
			Prompt:          prompt,
			Response:        content,
			TokensUsed:      tokensUsed,
			Temperature:     temperature,
			MaxTokens:       maxTokens,
			SimilarityScore: 1.0,
			Metadata: map[string]string{
				"user_id":    AnonymousUserID,
				"session_id": DefaultSessionID,
				"source":     "inference",
			},
			Tags: []string{domain, "inference", "generated"},
		}

		go func() {
			if err := s.semanticCache.Set(context.Background(), semanticEntry); err != nil {
				log.Printf("⚠️ Failed to cache semantic response: %v", err)
			}
		}()
	}
}

