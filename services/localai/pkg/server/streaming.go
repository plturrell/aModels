package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
)

// StreamingResponse represents a streaming response chunk
type StreamingResponse struct {
	ID       string                 `json:"id"`
	Object   string                 `json:"object"`
	Created  int64                  `json:"created"`
	Model    string                 `json:"model"`
	Choices  []StreamingChoice      `json:"choices"`
	Usage    *TokenUsage            `json:"usage,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// StreamingChoice represents a choice in streaming response
type StreamingChoice struct {
	Index        int               `json:"index"`
	Delta        *StreamingDelta   `json:"delta,omitempty"`
	Message      *StreamingMessage `json:"message,omitempty"`
	FinishReason *string           `json:"finish_reason,omitempty"`
}

// StreamingDelta represents a delta in streaming response
type StreamingDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// StreamingMessage represents a complete message in streaming response
type StreamingMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// TokenUsage represents token usage information
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamWriter handles streaming responses
type StreamWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	ctx     context.Context
}

// NewStreamWriter creates a new stream writer
func NewStreamWriter(w http.ResponseWriter, ctx context.Context) (*StreamWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported")
	}

	// Set headers for Server-Sent Events
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")

	return &StreamWriter{
		w:       w,
		flusher: flusher,
		ctx:     ctx,
	}, nil
}

// WriteChunk writes a streaming chunk
func (sw *StreamWriter) WriteChunk(data interface{}) error {
	select {
	case <-sw.ctx.Done():
		return sw.ctx.Err()
	default:
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal chunk: %w", err)
	}

	_, err = fmt.Fprintf(sw.w, "data: %s\n\n", jsonData)
	if err != nil {
		return fmt.Errorf("failed to write chunk: %w", err)
	}

	sw.flusher.Flush()
	return nil
}

// WriteError writes an error chunk
func (sw *StreamWriter) WriteError(err error) {
	errorChunk := map[string]interface{}{
		"error": map[string]interface{}{
			"message": err.Error(),
			"type":    "server_error",
		},
	}
	sw.WriteChunk(errorChunk)
}

// WriteDone writes the final done chunk
func (sw *StreamWriter) WriteDone() {
	_, _ = fmt.Fprintf(sw.w, "data: [DONE]\n\n")
	sw.flusher.Flush()
}

// StreamingInferenceRequest represents a streaming inference request
type StreamingInferenceRequest struct {
	Prompt      string
	Domain      string
	MaxTokens   int
	Temperature float64
	Model       *ai.VaultGemma
	TopP        float64
	TopK        int
}

// StreamingInferenceResponse represents a streaming inference response
type StreamingInferenceResponse struct {
	Content    string
	TokensUsed int
	Error      error
	Done       bool
}

// StreamInference performs streaming inference
func (s *VaultGemmaServer) StreamInference(ctx context.Context, req *StreamingInferenceRequest) <-chan StreamingInferenceResponse {
	responseChan := make(chan StreamingInferenceResponse, 10)

	go func() {
		defer close(responseChan)

		// Use enhanced inference engine if available
		if s.enhancedEngine != nil {
			enhancedReq := &inference.EnhancedInferenceRequest{
				Prompt:      req.Prompt,
				Domain:      req.Domain,
				MaxTokens:   req.MaxTokens,
				Temperature: req.Temperature,
				Model:       req.Model,
				TopP:        req.TopP,
				TopK:        req.TopK,
			}

			// Stream the response
			s.streamEnhancedResponse(ctx, enhancedReq, responseChan)
		} else {
			// Fall back to basic inference with simulated streaming
			s.streamBasicResponse(ctx, req, responseChan)
		}
	}()

	return responseChan
}

// streamEnhancedResponse streams enhanced inference response
func (s *VaultGemmaServer) streamEnhancedResponse(ctx context.Context, req *inference.EnhancedInferenceRequest, responseChan chan<- StreamingInferenceResponse) {
	// For now, we'll simulate streaming by chunking the response
	// In a real implementation, this would stream tokens as they're generated
	response := s.enhancedEngine.GenerateEnhancedResponse(ctx, req)

	if response.Error != nil {
		responseChan <- StreamingInferenceResponse{
			Error: response.Error,
			Done:  true,
		}
		return
	}

	// Simulate streaming by sending chunks
	content := response.Content
	chunkSize := 10 // Characters per chunk

	for i := 0; i < len(content); i += chunkSize {
		end := i + chunkSize
		if end > len(content) {
			end = len(content)
		}

		chunk := content[i:end]

		select {
		case <-ctx.Done():
			return
		case responseChan <- StreamingInferenceResponse{
			Content:    chunk,
			TokensUsed: 0, // Will be calculated at the end
			Done:       false,
		}:
		}

		// Simulate processing time
		time.Sleep(50 * time.Millisecond)
	}

	// Send final response with token count
	responseChan <- StreamingInferenceResponse{
		Content:    "",
		TokensUsed: response.TokensUsed,
		Done:       true,
	}
}

// streamBasicResponse streams basic inference response
func (s *VaultGemmaServer) streamBasicResponse(ctx context.Context, req *StreamingInferenceRequest, responseChan chan<- StreamingInferenceResponse) {
	if s.inferenceEngine == nil {
		s.inferenceEngine = inference.NewInferenceEngine(s.models, s.domainManager)
	}

	inferenceReq := &inference.InferenceRequest{
		Prompt:      req.Prompt,
		Domain:      req.Domain,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Model:       req.Model,
	}

	response := s.inferenceEngine.GenerateResponse(ctx, inferenceReq)

	if response.Error != nil {
		responseChan <- StreamingInferenceResponse{
			Error: response.Error,
			Done:  true,
		}
		return
	}

	// Simulate streaming by chunking the response
	content := response.Content
	chunkSize := 10 // Characters per chunk

	for i := 0; i < len(content); i += chunkSize {
		end := i + chunkSize
		if end > len(content) {
			end = len(content)
		}

		chunk := content[i:end]

		select {
		case <-ctx.Done():
			return
		case responseChan <- StreamingInferenceResponse{
			Content:    chunk,
			TokensUsed: 0, // Will be calculated at the end
			Done:       false,
		}:
		}

		// Simulate processing time
		time.Sleep(50 * time.Millisecond)
	}

	// Send final response with token count
	responseChan <- StreamingInferenceResponse{
		Content:    "",
		TokensUsed: response.TokensUsed,
		Done:       true,
	}
}

// HandleStreamingChat handles streaming chat requests
func (s *VaultGemmaServer) HandleStreamingChat(w http.ResponseWriter, r *http.Request) {
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

	// Create context with timeout
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second) // Longer timeout for streaming
	defer cancel()

	// Create stream writer
	streamWriter, err := NewStreamWriter(w, ctx)
	if err != nil {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

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
		MaxTokens   int      `json:"max_tokens,omitempty"`
		Temperature float64  `json:"temperature,omitempty"`
		TopP        float64  `json:"top_p,omitempty"`
		TopK        int      `json:"top_k,omitempty"`
		Domains     []string `json:"domains,omitempty"`
		Stream      bool     `json:"stream,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("Failed to decode request: %v", err)
		streamWriter.WriteError(fmt.Errorf("invalid JSON: %v", err))
		return
	}

	// Validate request
	if len(req.Messages) == 0 {
		streamWriter.WriteError(fmt.Errorf("messages array cannot be empty"))
		return
	}

	// Build prompt
	var prompt string
	for _, msg := range req.Messages {
		prompt += msg.Content + "\n"
	}

	// Detect or use specified domain
	domain := req.Model
	if domain == "auto" || domain == "" {
		domain = s.domainManager.DetectDomain(prompt, req.Domains)
		log.Printf("Auto-detected domain: %s", domain)
	}

	// Get domain configuration
	domainConfig, _ := s.domainManager.GetDomainConfig(domain)

	topP, topK := resolveSampling(req.TopP, req.TopK, domainConfig)
	req.TopP = topP
	req.TopK = topK

	// Resolve model
	modelKey := domain
	model, exists := s.models[modelKey]
	if !exists {
		if domainConfig != nil && domainConfig.FallbackModel != "" {
			if fallbackModel, ok := s.models[domainConfig.FallbackModel]; ok {
				model = fallbackModel
				modelKey = domainConfig.FallbackModel
			}
		}
	}

	if model == nil {
		defaultDomain := s.domainManager.GetDefaultDomain()
		if defaultDomain != "" {
			if defaultModel, ok := s.models[defaultDomain]; ok {
				model = defaultModel
				domain = defaultDomain
				modelKey = defaultDomain
				domainConfig, _ = s.domainManager.GetDomainConfig(defaultDomain)
			}
		}
		if model == nil {
			streamWriter.WriteError(fmt.Errorf("no models available"))
			return
		}
	}

	// Use domain-specific parameters
	maxTokens := req.MaxTokens
	if maxTokens == 0 && domainConfig != nil {
		maxTokens = domainConfig.MaxTokens
	}

	// Check for cached response first (exact cache)
	var cachedContent string
	var cachedTokens int
	cacheHit := false
	semanticHit := false

	if s.hanaCache != nil {
		cacheKey := fmt.Sprintf("streaming:%s:%s:%s", modelKey, domain, prompt)
		if cached, err := s.hanaCache.Get(ctx, cacheKey); err == nil && cached != nil && cached.Response != "" {
			cachedContent = cached.Response
			cachedTokens = cached.TokensUsed
			cacheHit = true
			log.Printf("📦 Cache hit for streaming domain: %s", domain)
		}
	}

	// If no exact cache hit, try semantic cache
	if !cacheHit && s.semanticCache != nil {
		// Find semantically similar cached responses
		similarEntries, err := s.semanticCache.FindSemanticSimilar(ctx, prompt, modelKey, domain, 0.8, 5)
		if err == nil && len(similarEntries) > 0 {
			// Use the most similar entry
			bestEntry := similarEntries[0]
			cachedContent = bestEntry.Response
			cachedTokens = bestEntry.TokensUsed
			semanticHit = true
			log.Printf("🧠 Semantic cache hit for streaming domain: %s (similarity: %.2f)", domain, bestEntry.SimilarityScore)
		}
	}

	// If we have a cached response, stream it instead of generating
	if cacheHit || semanticHit {
		// Stream the cached content
		responseID := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
		created := time.Now().Unix()

		// Send initial response
		initialResponse := StreamingResponse{
			ID:      responseID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   "vaultgemma",
			Choices: []StreamingChoice{
				{
					Index: 0,
					Delta: &StreamingDelta{
						Role: "assistant",
					},
				},
			},
		}

		if err := streamWriter.WriteChunk(initialResponse); err != nil {
			log.Printf("Failed to write initial chunk: %v", err)
			return
		}

		// Stream cached content in chunks
		chunkSize := 50 // characters per chunk
		for i := 0; i < len(cachedContent); i += chunkSize {
			end := i + chunkSize
			if end > len(cachedContent) {
				end = len(cachedContent)
			}
			chunk := cachedContent[i:end]

			chunkResponse := StreamingResponse{
				ID:      responseID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   "vaultgemma",
				Choices: []StreamingChoice{
					{
						Index: 0,
						Delta: &StreamingDelta{
							Content: chunk,
						},
					},
				},
			}

			if err := streamWriter.WriteChunk(chunkResponse); err != nil {
				log.Printf("Failed to write content chunk: %v", err)
				return
			}

			// Small delay to simulate streaming
			time.Sleep(10 * time.Millisecond)
		}

		// Send final response
		finalResponse := StreamingResponse{
			ID:      responseID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   "vaultgemma",
			Choices: []StreamingChoice{
				{
					Index: 0,
					Delta: &StreamingDelta{},
				},
			},
			Usage: &TokenUsage{
				PromptTokens:     len(strings.Split(prompt, " ")),
				CompletionTokens: cachedTokens,
				TotalTokens:      len(strings.Split(prompt, " ")) + cachedTokens,
			},
		}

		if err := streamWriter.WriteChunk(finalResponse); err != nil {
			log.Printf("Failed to write final chunk: %v", err)
			return
		}

		// Send done signal
		doneResponse := StreamingResponse{
			ID:      responseID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   "vaultgemma",
			Choices: []StreamingChoice{
				{
					Index:        0,
					Delta:        &StreamingDelta{},
					FinishReason: stringPtr("stop"),
				},
			},
			Usage: &TokenUsage{
				PromptTokens:     len(strings.Split(prompt, " ")),
				CompletionTokens: cachedTokens,
				TotalTokens:      len(strings.Split(prompt, " ")) + cachedTokens,
			},
		}

		if err := streamWriter.WriteChunk(doneResponse); err != nil {
			log.Printf("Failed to write done chunk: %v", err)
			return
		}

		return
	}

	// Create streaming inference request
	streamReq := &StreamingInferenceRequest{
		Prompt:      prompt,
		Domain:      domain,
		MaxTokens:   maxTokens,
		Temperature: req.Temperature,
		Model:       model,
		TopP:        topP,
		TopK:        topK,
	}

	// Start streaming
	responseID := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	created := time.Now().Unix()

	// Send initial response
	initialResponse := StreamingResponse{
		ID:      responseID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   "vaultgemma",
		Choices: []StreamingChoice{
			{
				Index: 0,
				Delta: &StreamingDelta{
					Role: "assistant",
				},
			},
		},
	}

	if err := streamWriter.WriteChunk(initialResponse); err != nil {
		log.Printf("Failed to write initial chunk: %v", err)
		return
	}

	// Stream the response
	responseChan := s.StreamInference(ctx, streamReq)
	var fullContent string
	var totalTokens int

	for response := range responseChan {
		if response.Error != nil {
			streamWriter.WriteError(response.Error)
			return
		}

		if response.Done {
			// Send final response with usage
			finalResponse := StreamingResponse{
				ID:      responseID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   "vaultgemma",
				Choices: []StreamingChoice{
					{
						Index:        0,
						FinishReason: stringPtr("stop"),
					},
				},
				Usage: &TokenUsage{
					PromptTokens:     len(prompt) / 4,
					CompletionTokens: response.TokensUsed,
					TotalTokens:      (len(prompt) / 4) + response.TokensUsed,
				},
			}

			if err := streamWriter.WriteChunk(finalResponse); err != nil {
				log.Printf("Failed to write final chunk: %v", err)
			}

			streamWriter.WriteDone()
			break
		}

		if response.Content != "" {
			fullContent += response.Content
			totalTokens += response.TokensUsed

			// Send content chunk
			contentResponse := StreamingResponse{
				ID:      responseID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   "vaultgemma",
				Choices: []StreamingChoice{
					{
						Index: 0,
						Delta: &StreamingDelta{
							Content: response.Content,
						},
					},
				},
			}

			if err := streamWriter.WriteChunk(contentResponse); err != nil {
				log.Printf("Failed to write content chunk: %v", err)
				return
			}
		}
	}

	// Cache the generated response for future use
	if fullContent != "" {
		// Store in HANA cache
		if s.hanaCache != nil {
			cacheKey := fmt.Sprintf("streaming:%s:%s:%s", modelKey, domain, prompt)
			cacheEntry := &storage.CacheEntry{
				CacheKey:    cacheKey,
				PromptHash:  fmt.Sprintf("%x", []byte(prompt)),
				Model:       modelKey,
				Domain:      domain,
				Response:    fullContent,
				TokensUsed:  totalTokens,
				Temperature: req.Temperature,
				MaxTokens:   maxTokens,
				CreatedAt:   time.Now(),
				ExpiresAt:   time.Now().Add(24 * time.Hour),
			}
			go func() {
				if err := s.hanaCache.Set(context.Background(), cacheEntry); err != nil {
					log.Printf("⚠️ Failed to cache streaming response: %v", err)
				}
			}()
		}

		// Store in semantic cache for similarity matching
		if s.semanticCache != nil {
			semanticCacheKey := s.semanticCache.GenerateCacheKey(prompt, modelKey, domain, req.Temperature, maxTokens, topP, topK)
			semanticEntry := &storage.SemanticCacheEntry{
				CacheKey:        semanticCacheKey,
				PromptHash:      fmt.Sprintf("%x", []byte(prompt)),
				SemanticHash:    s.semanticCache.GenerateSemanticHash(prompt),
				Model:           modelKey,
				Domain:          domain,
				Prompt:          prompt,
				Response:        fullContent,
				TokensUsed:      totalTokens,
				Temperature:     req.Temperature,
				MaxTokens:       maxTokens,
				SimilarityScore: 1.0,
				Metadata: map[string]string{
					"user_id":    "anonymous",
					"session_id": "default",
					"source":     "streaming",
				},
				Tags: []string{domain, "streaming", "generated"},
			}

			// Cache asynchronously
			go func() {
				if err := s.semanticCache.Set(context.Background(), semanticEntry); err != nil {
					log.Printf("⚠️ Failed to cache semantic streaming response: %v", err)
				}
			}()
		}
	}

	duration := time.Since(start)
	log.Printf("Streaming chat request processed in %.2fms for domain: %s", duration.Seconds()*1000, domain)
}

// Helper function to create string pointer
func stringPtr(s string) *string {
	return &s
}
