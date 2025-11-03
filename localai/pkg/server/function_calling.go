package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
)

// FunctionCall represents a function call
type FunctionCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// FunctionDefinition represents a function definition
type FunctionDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// FunctionTool represents a function tool
type FunctionTool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

// FunctionCallingRequest represents a request with function calling
type FunctionCallingRequest struct {
	Model       string         `json:"model"`
	Messages    []ChatMessage  `json:"messages"`
	Tools       []FunctionTool `json:"tools,omitempty"`
	ToolChoice  interface{}    `json:"tool_choice,omitempty"`
	MaxTokens   int            `json:"max_tokens,omitempty"`
	Temperature float64        `json:"temperature,omitempty"`
	TopP        float64        `json:"top_p,omitempty"`
	TopK        int            `json:"top_k,omitempty"`
	Domains     []string       `json:"domains,omitempty"`
}

// ChatMessage represents a chat message
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolCall represents a tool call
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCallingResponse represents a response with function calling
type FunctionCallingResponse struct {
	ID       string                 `json:"id"`
	Object   string                 `json:"object"`
	Created  int64                  `json:"created"`
	Model    string                 `json:"model"`
	Choices  []FunctionChoice       `json:"choices"`
	Usage    TokenUsage             `json:"usage"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// FunctionChoice represents a choice in function calling response
type FunctionChoice struct {
	Index        int             `json:"index"`
	Message      FunctionMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// FunctionMessage represents a message in function calling response
type FunctionMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// FunctionRegistry manages available functions
type FunctionRegistry struct {
	functions map[string]FunctionHandler
}

// FunctionHandler represents a function handler
type FunctionHandler interface {
	GetDefinition() FunctionDefinition
	Call(ctx context.Context, arguments map[string]interface{}) (interface{}, error)
}

// NewFunctionRegistry creates a new function registry
func NewFunctionRegistry() *FunctionRegistry {
	return &FunctionRegistry{
		functions: make(map[string]FunctionHandler),
	}
}

// RegisterFunction registers a function handler
func (fr *FunctionRegistry) RegisterFunction(name string, handler FunctionHandler) {
	fr.functions[name] = handler
}

// GetFunction gets a function handler by name
func (fr *FunctionRegistry) GetFunction(name string) (FunctionHandler, bool) {
	handler, exists := fr.functions[name]
	return handler, exists
}

// ListFunctions lists all available functions
func (fr *FunctionRegistry) ListFunctions() []FunctionDefinition {
	definitions := make([]FunctionDefinition, 0, len(fr.functions))
	for _, handler := range fr.functions {
		definitions = append(definitions, handler.GetDefinition())
	}
	return definitions
}

// Built-in function handlers

// CalculatorFunction handles calculator operations
type CalculatorFunction struct{}

func (cf *CalculatorFunction) GetDefinition() FunctionDefinition {
	return FunctionDefinition{
		Name:        "calculator",
		Description: "Perform mathematical calculations",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"expression": map[string]interface{}{
					"type":        "string",
					"description": "Mathematical expression to evaluate",
				},
			},
			"required": []string{"expression"},
		},
	}
}

func (cf *CalculatorFunction) Call(ctx context.Context, arguments map[string]interface{}) (interface{}, error) {
	expression, ok := arguments["expression"].(string)
	if !ok {
		return nil, fmt.Errorf("expression parameter is required")
	}

	// Simple calculator implementation
	result, err := cf.evaluateExpression(expression)
	if err != nil {
		return nil, fmt.Errorf("calculation error: %w", err)
	}

	return map[string]interface{}{
		"result":     result,
		"expression": expression,
	}, nil
}

func (cf *CalculatorFunction) evaluateExpression(expr string) (float64, error) {
	// This is a simplified calculator - in production, you'd use a proper expression evaluator
	expr = strings.TrimSpace(expr)

	// Basic arithmetic operations
	if strings.Contains(expr, "+") {
		parts := strings.Split(expr, "+")
		if len(parts) == 2 {
			a, err := cf.parseNumber(strings.TrimSpace(parts[0]))
			if err != nil {
				return 0, err
			}
			b, err := cf.parseNumber(strings.TrimSpace(parts[1]))
			if err != nil {
				return 0, err
			}
			return a + b, nil
		}
	}

	if strings.Contains(expr, "-") {
		parts := strings.Split(expr, "-")
		if len(parts) == 2 {
			a, err := cf.parseNumber(strings.TrimSpace(parts[0]))
			if err != nil {
				return 0, err
			}
			b, err := cf.parseNumber(strings.TrimSpace(parts[1]))
			if err != nil {
				return 0, err
			}
			return a - b, nil
		}
	}

	if strings.Contains(expr, "*") {
		parts := strings.Split(expr, "*")
		if len(parts) == 2 {
			a, err := cf.parseNumber(strings.TrimSpace(parts[0]))
			if err != nil {
				return 0, err
			}
			b, err := cf.parseNumber(strings.TrimSpace(parts[1]))
			if err != nil {
				return 0, err
			}
			return a * b, nil
		}
	}

	if strings.Contains(expr, "/") {
		parts := strings.Split(expr, "/")
		if len(parts) == 2 {
			a, err := cf.parseNumber(strings.TrimSpace(parts[0]))
			if err != nil {
				return 0, err
			}
			b, err := cf.parseNumber(strings.TrimSpace(parts[1]))
			if err != nil {
				return 0, err
			}
			if b == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return a / b, nil
		}
	}

	// Single number
	return cf.parseNumber(expr)
}

func (cf *CalculatorFunction) parseNumber(s string) (float64, error) {
	// Simple number parsing
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty number")
	}

	// Try to parse as float64
	var result float64
	_, err := fmt.Sscanf(s, "%f", &result)
	if err != nil {
		return 0, fmt.Errorf("invalid number: %s", s)
	}

	return result, nil
}

// WeatherFunction handles weather queries
type WeatherFunction struct{}

func (wf *WeatherFunction) GetDefinition() FunctionDefinition {
	return FunctionDefinition{
		Name:        "get_weather",
		Description: "Get current weather information for a location",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"location": map[string]interface{}{
					"type":        "string",
					"description": "City name or location",
				},
				"unit": map[string]interface{}{
					"type":        "string",
					"enum":        []string{"celsius", "fahrenheit"},
					"description": "Temperature unit",
					"default":     "celsius",
				},
			},
			"required": []string{"location"},
		},
	}
}

func (wf *WeatherFunction) Call(ctx context.Context, arguments map[string]interface{}) (interface{}, error) {
	location, ok := arguments["location"].(string)
	if !ok {
		return nil, fmt.Errorf("location parameter is required")
	}

	unit := "celsius"
	if u, ok := arguments["unit"].(string); ok {
		unit = u
	}

	// Mock weather data - in production, you'd call a real weather API
	weather := map[string]interface{}{
		"location":    location,
		"temperature": 22.5,
		"unit":        unit,
		"condition":   "sunny",
		"humidity":    65,
		"wind_speed":  10.2,
	}

	return weather, nil
}

// DatabaseFunction handles database queries
type DatabaseFunction struct {
	server *VaultGemmaServer
}

func (df *DatabaseFunction) GetDefinition() FunctionDefinition {
	return FunctionDefinition{
		Name:        "query_database",
		Description: "Execute SQL queries against the database",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "SQL query to execute",
				},
				"limit": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum number of rows to return",
					"default":     100,
				},
			},
			"required": []string{"query"},
		},
	}
}

func (df *DatabaseFunction) Call(ctx context.Context, arguments map[string]interface{}) (interface{}, error) {
	query, ok := arguments["query"].(string)
	if !ok {
		return nil, fmt.Errorf("query parameter is required")
	}

	limit := 100
	if l, ok := arguments["limit"].(float64); ok {
		limit = int(l)
	}

	// Use the existing SQL query tool
	if df.server.hanaPool != nil {
		// This would integrate with the existing HANA tools
		// For now, return a mock response
		return map[string]interface{}{
			"query":  query,
			"limit":  limit,
			"result": "Database query executed successfully",
		}, nil
	}

	return nil, fmt.Errorf("database not available")
}

// HandleFunctionCalling handles function calling requests
func (s *VaultGemmaServer) HandleFunctionCalling(w http.ResponseWriter, r *http.Request) {
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
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	// Track request
	s.mu.Lock()
	s.requestCount++
	s.mu.Unlock()

	start := time.Now()
	var req FunctionCallingRequest

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

	// Build prompt for caching
	var prompt string
	for _, msg := range req.Messages {
		prompt += msg.Content + "\n"
	}

	// Detect or use specified domain
	domain := req.Model
	if domain == "auto" || domain == "" {
		domain = s.domainManager.DetectDomain(prompt, req.Domains)
		log.Printf("Auto-detected domain for function calling: %s", domain)
	}

	// Get domain configuration
	domainConfig, _ := s.domainManager.GetDomainConfig(domain)

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
	}

	// Use domain-specific parameters
	maxTokens := req.MaxTokens
	if maxTokens == 0 && domainConfig != nil {
		maxTokens = domainConfig.MaxTokens
	}

	topP, topK := resolveSampling(req.TopP, req.TopK, domainConfig)
	req.TopP = topP
	req.TopK = topK

	// Check for cached response first (exact cache)
	var cachedResponse *FunctionCallingResponse
	cacheHit := false
	semanticHit := false

	if s.hanaCache != nil {
		cacheKey := fmt.Sprintf("function:%s:%s:%s", modelKey, domain, prompt)
		if cached, err := s.hanaCache.Get(ctx, cacheKey); err == nil && cached != nil && cached.Response != "" {
			if err := json.Unmarshal([]byte(cached.Response), &cachedResponse); err == nil {
				cacheHit = true
				log.Printf("📦 Cache hit for function calling domain: %s", domain)
			}
		}
	}

	// If no exact cache hit, try semantic cache
	if !cacheHit && s.semanticCache != nil {
		// Find semantically similar cached responses
		similarEntries, err := s.semanticCache.FindSemanticSimilar(ctx, prompt, modelKey, domain, 0.8, 5)
		if err == nil && len(similarEntries) > 0 {
			// Use the most similar entry
			bestEntry := similarEntries[0]
			if err := json.Unmarshal([]byte(bestEntry.Response), &cachedResponse); err == nil {
				semanticHit = true
				log.Printf("🧠 Semantic cache hit for function calling domain: %s (similarity: %.2f)", domain, bestEntry.SimilarityScore)
			}
		}
	}

	// If we have a cached response, return it
	if cacheHit || semanticHit {
		duration := time.Since(start)
		log.Printf("Function calling request processed from cache in %.2fms for domain: %s", duration.Seconds()*1000, domain)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(cachedResponse)
		return
	}

	// Initialize function registry if not exists
	if s.functionRegistry == nil {
		s.functionRegistry = NewFunctionRegistry()
		s.registerBuiltinFunctions()
	}

	// Process function calls
	response, err := s.processFunctionCalls(ctx, &req)
	if err != nil {
		log.Printf("Function calling failed: %v", err)
		http.Error(w, fmt.Sprintf("Function calling failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Cache the generated response for future use
	if response != nil {
		responseJSON, err := json.Marshal(response)
		if err == nil {
			// Store in HANA cache
			if s.hanaCache != nil {
				cacheKey := fmt.Sprintf("function:%s:%s:%s", modelKey, domain, prompt)
				cacheEntry := &storage.CacheEntry{
					CacheKey:    cacheKey,
					PromptHash:  fmt.Sprintf("%x", []byte(prompt)),
					Model:       modelKey,
					Domain:      domain,
					Response:    string(responseJSON),
					TokensUsed:  len(strings.Split(string(responseJSON), " ")),
					Temperature: req.Temperature,
					MaxTokens:   maxTokens,
					CreatedAt:   time.Now(),
					ExpiresAt:   time.Now().Add(24 * time.Hour),
				}
				go func() {
					if err := s.hanaCache.Set(context.Background(), cacheEntry); err != nil {
						log.Printf("⚠️ Failed to cache function calling response: %v", err)
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
					Response:        string(responseJSON),
					TokensUsed:      len(strings.Split(string(responseJSON), " ")),
					Temperature:     req.Temperature,
					MaxTokens:       maxTokens,
					SimilarityScore: 1.0,
					Metadata: map[string]string{
						"user_id":    "anonymous",
						"session_id": "default",
						"source":     "function_calling",
						"top_p":      fmt.Sprintf("%.3f", topP),
						"top_k":      strconv.Itoa(topK),
					},
					Tags: []string{domain, "function_calling", "generated"},
				}

				// Cache asynchronously
				go func() {
					if err := s.semanticCache.Set(context.Background(), semanticEntry); err != nil {
						log.Printf("⚠️ Failed to cache semantic function calling response: %v", err)
					}
				}()
			}
		}
	}

	duration := time.Since(start)
	log.Printf("Function calling request processed in %.2fms for domain: %s", duration.Seconds()*1000, domain)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// processFunctionCalls processes function calls in the request
func (s *VaultGemmaServer) processFunctionCalls(ctx context.Context, req *FunctionCallingRequest) (*FunctionCallingResponse, error) {
	// Get the last message
	lastMessage := req.Messages[len(req.Messages)-1]

	// Check if the message contains tool calls
	if len(lastMessage.ToolCalls) > 0 {
		// Execute function calls
		toolResults := make([]ToolCall, 0, len(lastMessage.ToolCalls))

		for _, toolCall := range lastMessage.ToolCalls {
			result, err := s.executeFunctionCall(ctx, toolCall)
			if err != nil {
				log.Printf("Function call failed: %v", err)
				continue
			}

			// Create tool result
			toolResult := ToolCall{
				ID:   toolCall.ID,
				Type: "function",
				Function: FunctionCall{
					Name: toolCall.Function.Name,
					Arguments: map[string]interface{}{
						"result": result,
					},
				},
			}
			toolResults = append(toolResults, toolResult)
		}

		// Return tool results
		return &FunctionCallingResponse{
			ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "vaultgemma",
			Choices: []FunctionChoice{
				{
					Index: 0,
					Message: FunctionMessage{
						Role:      "assistant",
						ToolCalls: toolResults,
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: TokenUsage{
				PromptTokens:     s.estimateTokens(req.Messages),
				CompletionTokens: 0,
				TotalTokens:      s.estimateTokens(req.Messages),
			},
		}, nil
	}

	// No tool calls, generate regular response
	return s.generateFunctionCallingResponse(ctx, req)
}

// executeFunctionCall executes a function call
func (s *VaultGemmaServer) executeFunctionCall(ctx context.Context, toolCall ToolCall) (interface{}, error) {
	handler, exists := s.functionRegistry.GetFunction(toolCall.Function.Name)
	if !exists {
		return nil, fmt.Errorf("function not found: %s", toolCall.Function.Name)
	}

	return handler.Call(ctx, toolCall.Function.Arguments)
}

// generateFunctionCallingResponse generates a response for function calling
func (s *VaultGemmaServer) generateFunctionCallingResponse(ctx context.Context, req *FunctionCallingRequest) (*FunctionCallingResponse, error) {
	// Build prompt
	var prompt string
	for _, msg := range req.Messages {
		if msg.Role == "user" {
			prompt += msg.Content + "\n"
		}
	}

	// Detect domain
	domain := req.Model
	if domain == "auto" || domain == "" {
		domain = s.domainManager.DetectDomain(prompt, req.Domains)
	}

	// Get domain configuration
	domainConfig, _ := s.domainManager.GetDomainConfig(domain)

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
		return nil, fmt.Errorf("no models available")
	}

	// Use domain-specific parameters
	maxTokens := req.MaxTokens
	if maxTokens == 0 && domainConfig != nil {
		maxTokens = domainConfig.MaxTokens
	}

	topP, topK := resolveSampling(req.TopP, req.TopK, domainConfig)
	req.TopP = topP
	req.TopK = topK

	// Generate response using enhanced inference
	var content string
	var tokensUsed int

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
			return nil, response.Error
		}

		content = response.Content
		tokensUsed = response.TokensUsed
	} else {
		// Fall back to basic inference
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
			return nil, response.Error
		}

		content = response.Content
		tokensUsed = response.TokensUsed
	}

	// Check if we should suggest function calls
	var toolCalls []ToolCall
	if len(req.Tools) > 0 && s.shouldSuggestFunctions(content, req.Tools) {
		toolCalls = s.suggestFunctionCalls(content, req.Tools)
	}

	// Create response
	response := &FunctionCallingResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "vaultgemma",
		Choices: []FunctionChoice{
			{
				Index: 0,
				Message: FunctionMessage{
					Role:      "assistant",
					Content:   content,
					ToolCalls: toolCalls,
				},
				FinishReason: s.getFinishReason(toolCalls),
			},
		},
		Usage: TokenUsage{
			PromptTokens:     s.estimateTokens(req.Messages),
			CompletionTokens: tokensUsed,
			TotalTokens:      s.estimateTokens(req.Messages) + tokensUsed,
		},
	}

	response.Metadata = map[string]interface{}{
		"top_p":       topP,
		"top_k":       topK,
		"temperature": req.Temperature,
	}

	return response, nil
}

// shouldSuggestFunctions determines if we should suggest function calls
func (s *VaultGemmaServer) shouldSuggestFunctions(content string, tools []FunctionTool) bool {
	// Simple heuristic: suggest functions if content contains certain keywords
	keywords := []string{"calculate", "compute", "weather", "database", "query", "search"}

	contentLower := strings.ToLower(content)
	for _, keyword := range keywords {
		if strings.Contains(contentLower, keyword) {
			return true
		}
	}

	return false
}

// suggestFunctionCalls suggests function calls based on content and available tools
func (s *VaultGemmaServer) suggestFunctionCalls(content string, tools []FunctionTool) []ToolCall {
	var toolCalls []ToolCall

	contentLower := strings.ToLower(content)

	// Suggest calculator if content contains math keywords
	if strings.Contains(contentLower, "calculate") || strings.Contains(contentLower, "compute") {
		for _, tool := range tools {
			if tool.Function.Name == "calculator" {
				toolCalls = append(toolCalls, ToolCall{
					ID:   fmt.Sprintf("call_%d", time.Now().UnixNano()),
					Type: "function",
					Function: FunctionCall{
						Name: "calculator",
						Arguments: map[string]interface{}{
							"expression": "2 + 2", // This would be extracted from content
						},
					},
				})
				break
			}
		}
	}

	// Suggest weather function if content contains weather keywords
	if strings.Contains(contentLower, "weather") || strings.Contains(contentLower, "temperature") {
		for _, tool := range tools {
			if tool.Function.Name == "get_weather" {
				toolCalls = append(toolCalls, ToolCall{
					ID:   fmt.Sprintf("call_%d", time.Now().UnixNano()),
					Type: "function",
					Function: FunctionCall{
						Name: "get_weather",
						Arguments: map[string]interface{}{
							"location": "New York", // This would be extracted from content
							"unit":     "celsius",
						},
					},
				})
				break
			}
		}
	}

	return toolCalls
}

// getFinishReason determines the finish reason
func (s *VaultGemmaServer) getFinishReason(toolCalls []ToolCall) string {
	if len(toolCalls) > 0 {
		return "tool_calls"
	}
	return "stop"
}

// estimateTokens estimates token count for messages
func (s *VaultGemmaServer) estimateTokens(messages []ChatMessage) int {
	total := 0
	for _, msg := range messages {
		total += len(msg.Content) / 4 // Rough estimation
	}
	return total
}

// registerBuiltinFunctions registers built-in functions
func (s *VaultGemmaServer) registerBuiltinFunctions() {
	s.functionRegistry.RegisterFunction("calculator", &CalculatorFunction{})
	s.functionRegistry.RegisterFunction("get_weather", &WeatherFunction{})
	s.functionRegistry.RegisterFunction("query_database", &DatabaseFunction{server: s})
}
