// Copyright 2025 AgenticAI ETH Contributors
// SPDX-License-Identifier: MIT

// Package client provides the consolidated LocalAI integration for all layers
//
// This replaces all LocalAI implementations across Layer 1 and Layer 4:
// - Layer 1: infrastructure/ai/localai_client.go
// - Layer 4: agenticAiETH_layer4_LocalAI/pkg/server/enhanced_localai_server.go
//
// Architecture: Unified | Component: consolidated-localai-client
// Version: 3.0.0 | Dependencies: tracked
// Last-Modified: 2025-01-27T12:00:00Z

package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/log"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/types"
)

// ConsolidatedLocalAIClient provides unified LocalAI integration for all layers
type ConsolidatedLocalAIClient struct {
	// Core configuration
	endpoints    []string
	currentIndex int
	httpClient   *http.Client
	config       *LocalAIConfig

	// Enhanced features
	circuitBreaker *CircuitBreaker
	retryConfig    *RetryConfig
	fallbackMode   bool
	cache          *SemanticCache
	loadBalancer   *ConsolidatedLoadBalancer

	// Layer-specific features
	layer1Features *Layer1LocalAIFeatures
	layer4Features *Layer4LocalAIFeatures

	// State management
	mu              sync.RWMutex
	initialized     bool
	healthStatus    string
	lastHealthCheck time.Time
}

// LocalAIConfig holds configuration for the consolidated client
type LocalAIConfig struct {
	// Basic settings
	Endpoints  []string
	ModelName  string
	APIKey     string
	Timeout    time.Duration
	MaxRetries int

	// Enhanced features
	EnableCircuitBreaker bool
	EnableSemanticCache  bool
	EnableLoadBalancing  bool
	EnableFallback       bool

	// Circuit breaker settings
	FailureThreshold int
	SuccessThreshold int
	CircuitTimeout   time.Duration

	// Cache settings
	CacheSize           int
	CacheTTL            time.Duration
	SimilarityThreshold float64

	// Load balancing settings
	LoadBalancingStrategy string // "round_robin", "least_loaded", "lowest_latency"
	HealthCheckInterval   time.Duration

	// Layer-specific settings
	EnableLayer1 bool
	EnableLayer4 bool
}

// CircuitBreaker implements circuit breaker pattern
type CircuitBreaker struct {
	failureCount     int
	successCount     int
	failureThreshold int
	successThreshold int
	timeout          time.Duration
	lastFailureTime  time.Time
	state            CircuitState
	mu               sync.RWMutex
}

// CircuitState represents the state of the circuit breaker
type CircuitState int

const (
	StateClosed CircuitState = iota
	StateOpen
	StateHalfOpen
)

// RetryConfig holds retry configuration
type RetryConfig struct {
	MaxAttempts int
	BaseDelay   time.Duration
	MaxDelay    time.Duration
	Multiplier  float64
	Jitter      bool
}

// SemanticCache provides semantic caching capabilities
type SemanticCache struct {
	entries map[string]*CacheEntry
	mu      sync.RWMutex
	maxSize int
	ttl     time.Duration
}

// CacheEntry represents a cached entry
type CacheEntry struct {
	Content     string
	Embedding   []float32
	Timestamp   time.Time
	AccessCount int
	Tags        []string
}

// LoadBalancer provides load balancing capabilities
type ConsolidatedLoadBalancer struct {
	endpoints []string
	latencies map[string]time.Duration
	loads     map[string]int
	mu        sync.RWMutex
}

// Layer1LocalAIFeatures provides Layer 1 specific features
type Layer1LocalAIFeatures struct {
	reasoningEngine    *ReasoningEngine
	collaborativeAI    *CollaborativeAI
	modelRouter        *ModelRouter
	performanceTracker *PerformanceTracker
}

// Layer4LocalAIFeatures provides Layer 4 specific features
type Layer4LocalAIFeatures struct {
	enhancedServer     *EnhancedServer
	semanticCaching    *SemanticCaching
	intelligentRouting *IntelligentRouting
	fallbackMechanism  *FallbackMechanism
}

// NewConsolidatedLocalAIClient creates a new consolidated LocalAI client
func NewConsolidatedLocalAIClient(config *LocalAIConfig) (*ConsolidatedLocalAIClient, error) {
	if config == nil {
		config = DefaultLocalAIConfig()
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	client := &ConsolidatedLocalAIClient{
		endpoints:       config.Endpoints,
		config:          config,
		healthStatus:    "initializing",
		lastHealthCheck: time.Now(),
	}

	if err := client.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize consolidated LocalAI client: %w", err)
	}

	return client, nil
}

// DefaultLocalAIConfig returns default configuration
func DefaultLocalAIConfig() *LocalAIConfig {
	return &LocalAIConfig{
		Endpoints:  []string{"http://localhost:8080"},
		ModelName:  "phi-3.5-mini",
		APIKey:     "",
		Timeout:    60 * time.Second,
		MaxRetries: 3,

		EnableCircuitBreaker: true,
		EnableSemanticCache:  true,
		EnableLoadBalancing:  true,
		EnableFallback:       true,

		FailureThreshold: 5,
		SuccessThreshold: 3,
		CircuitTimeout:   30 * time.Second,

		CacheSize:           1000,
		CacheTTL:            24 * time.Hour,
		SimilarityThreshold: 0.8,

		LoadBalancingStrategy: "least_loaded",
		HealthCheckInterval:   30 * time.Second,

		EnableLayer1: true,
		EnableLayer4: true,
	}
}

// Validate validates the configuration
func (c *LocalAIConfig) Validate() error {
	if len(c.Endpoints) == 0 {
		return fmt.Errorf("at least one endpoint is required")
	}
	if c.ModelName == "" {
		return fmt.Errorf("model name is required")
	}
	if c.Timeout <= 0 {
		return fmt.Errorf("timeout must be positive")
	}
	if c.MaxRetries < 0 {
		return fmt.Errorf("max retries must be non-negative")
	}
	return nil
}

func (c *ConsolidatedLocalAIClient) initialize() error {
	// Initialize HTTP client
	c.httpClient = &http.Client{
		Timeout: c.config.Timeout,
	}

	// Initialize circuit breaker
	if c.config.EnableCircuitBreaker {
		c.circuitBreaker = &CircuitBreaker{
			failureThreshold: c.config.FailureThreshold,
			successThreshold: c.config.SuccessThreshold,
			timeout:          c.config.CircuitTimeout,
			state:            StateClosed,
		}
	}

	// Initialize retry config
	c.retryConfig = &RetryConfig{
		MaxAttempts: c.config.MaxRetries,
		BaseDelay:   100 * time.Millisecond,
		MaxDelay:    5 * time.Second,
		Multiplier:  2.0,
		Jitter:      true,
	}

	// Initialize semantic cache
	if c.config.EnableSemanticCache {
		c.cache = &SemanticCache{
			entries: make(map[string]*CacheEntry),
			maxSize: c.config.CacheSize,
			ttl:     c.config.CacheTTL,
		}
	}

	// Initialize load balancer
	if c.config.EnableLoadBalancing {
		c.loadBalancer = &ConsolidatedLoadBalancer{
			endpoints: c.config.Endpoints,
			latencies: make(map[string]time.Duration),
			loads:     make(map[string]int),
		}
	}

	// Initialize layer-specific features
	if err := c.initializeLayerFeatures(); err != nil {
		return fmt.Errorf("failed to initialize layer features: %w", err)
	}

	c.initialized = true
	c.healthStatus = "healthy"

	log.Info("Consolidated LocalAI client initialized successfully",
		"endpoints", len(c.config.Endpoints),
		"model", c.config.ModelName,
		"layer1", c.config.EnableLayer1,
		"layer4", c.config.EnableLayer4)

	return nil
}

func (c *ConsolidatedLocalAIClient) initializeLayerFeatures() error {
	// Initialize Layer 1 features
	if c.config.EnableLayer1 {
		c.layer1Features = &Layer1LocalAIFeatures{
			reasoningEngine:    NewReasoningEngine(),
			collaborativeAI:    NewCollaborativeAI(),
			modelRouter:        NewModelRouter(),
			performanceTracker: NewPerformanceTracker(),
		}
	}

	// Initialize Layer 4 features
	if c.config.EnableLayer4 {
		c.layer4Features = &Layer4LocalAIFeatures{
			enhancedServer:     NewEnhancedServer(c.config.Endpoints),
			semanticCaching:    NewSemanticCaching(c.cache),
			intelligentRouting: NewIntelligentRouting(),
			fallbackMechanism:  NewFallbackMechanism(),
		}
	}

	return nil
}

// Core API Methods

// Generate sends a generation request with enhanced features
func (c *ConsolidatedLocalAIClient) Generate(ctx context.Context, req *types.GenerateRequest) (*types.GenerateResponse, error) {
	// Check circuit breaker
	if c.circuitBreaker != nil && !c.circuitBreaker.CanExecute() {
		return c.handleFallback(ctx, req, "circuit_breaker_open")
	}

	// Check semantic cache
	if c.cache != nil {
		if cached, found := c.cache.Get(req.Prompt); found {
			return &types.GenerateResponse{
				Text:         cached.Content,
				TokensUsed:   0, // Cached response
				FinishReason: "cached",
				Metadata: map[string]interface{}{
					"cached":    true,
					"cache_hit": true,
				},
			}, nil
		}
	}

	// Process with retry logic
	result, err := c.processWithRetry(ctx, req)
	if err != nil {
		// Record failure
		if c.circuitBreaker != nil {
			c.circuitBreaker.RecordFailure()
		}

		// Try fallback
		fallbackResult, fallbackErr := c.handleFallback(ctx, req, err.Error())
		if fallbackErr != nil {
			return nil, fmt.Errorf("primary failed: %w, fallback failed: %w", err, fallbackErr)
		}

		return fallbackResult, nil
	}

	// Record success
	if c.circuitBreaker != nil {
		c.circuitBreaker.RecordSuccess()
	}

	// Cache result
	if c.cache != nil {
		c.cache.Set(req.Prompt, result.Text, []string{"generation"})
	}

	return result, nil
}

// GenerateChat sends a chat completion request
func (c *ConsolidatedLocalAIClient) GenerateChat(ctx context.Context, messages []types.ChatMessage, req *types.GenerateRequest) (*types.GenerateResponse, error) {
	// Convert to generation request
	prompt := c.buildChatPrompt(messages)
	genReq := &types.GenerateRequest{
		Prompt:           prompt,
		Temperature:      req.Temperature,
		MaxTokens:        req.MaxTokens,
		TopP:             req.TopP,
		FrequencyPenalty: req.FrequencyPenalty,
		PresencePenalty:  req.PresencePenalty,
		StopSequences:    req.StopSequences,
	}

	return c.Generate(ctx, genReq)
}

// EmbedText requests text embeddings
func (c *ConsolidatedLocalAIClient) EmbedText(ctx context.Context, embedModel string, input string) ([]float64, error) {
	if embedModel == "" {
		embedModel = c.config.ModelName
	}

	reqBody := map[string]interface{}{
		"model": embedModel,
		"input": input,
	}

	b, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal embeddings request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.getEndpoint()+"/v1/embeddings", bytes.NewBuffer(b))
	if err != nil {
		return nil, fmt.Errorf("create embeddings request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embeddings request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embeddings status %d: %s", resp.StatusCode, string(body))
	}

	var apiResp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("decode embeddings: %w", err)
	}

	if len(apiResp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return apiResp.Data[0].Embedding, nil
}

// Health and Status Methods

// HealthCheck verifies the LocalAI endpoint is accessible
func (c *ConsolidatedLocalAIClient) HealthCheck(ctx context.Context) error {
	endpoint := c.getEndpoint()
	req, err := http.NewRequestWithContext(ctx, "GET", endpoint+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	c.healthStatus = "healthy"
	c.lastHealthCheck = time.Now()
	return nil
}

// GetHealthStatus returns the current health status
func (c *ConsolidatedLocalAIClient) GetHealthStatus() string {
	return c.healthStatus
}

// GetMetrics returns current metrics
func (c *ConsolidatedLocalAIClient) GetMetrics() map[string]interface{} {
	metrics := map[string]interface{}{
		"health_status":     c.healthStatus,
		"last_health_check": c.lastHealthCheck,
		"endpoints":         len(c.config.Endpoints),
		"current_endpoint":  c.getEndpoint(),
	}

	if c.circuitBreaker != nil {
		metrics["circuit_breaker_state"] = c.circuitBreaker.GetState()
		metrics["failure_count"] = c.circuitBreaker.GetFailureCount()
		metrics["success_count"] = c.circuitBreaker.GetSuccessCount()
	}

	if c.cache != nil {
		metrics["cache_size"] = c.cache.Size()
		metrics["cache_hit_rate"] = c.cache.GetHitRate()
	}

	if c.loadBalancer != nil {
		metrics["load_balancer_strategy"] = c.config.LoadBalancingStrategy
		metrics["endpoint_loads"] = c.loadBalancer.GetLoads()
	}

	return metrics
}

// Close closes the client
func (c *ConsolidatedLocalAIClient) Close() error {
	if !c.initialized {
		return nil
	}

	// Close layer-specific features
	if c.layer1Features != nil {
		c.layer1Features.reasoningEngine.Close()
		c.layer1Features.collaborativeAI.Close()
		c.layer1Features.modelRouter.Close()
		c.layer1Features.performanceTracker.Close()
	}

	if c.layer4Features != nil {
		c.layer4Features.enhancedServer.Close()
		c.layer4Features.semanticCaching.Close()
		c.layer4Features.intelligentRouting.Close()
		c.layer4Features.fallbackMechanism.Close()
	}

	c.initialized = false
	c.healthStatus = "closed"

	log.Info("Consolidated LocalAI client closed")
	return nil
}

// Helper methods

func (c *ConsolidatedLocalAIClient) getEndpoint() string {
	if c.loadBalancer != nil {
		return c.loadBalancer.GetBestEndpoint()
	}
	return c.endpoints[c.currentIndex%len(c.endpoints)]
}

func (c *ConsolidatedLocalAIClient) processWithRetry(ctx context.Context, req *types.GenerateRequest) (*types.GenerateResponse, error) {
	var lastErr error

	for attempt := 0; attempt < c.retryConfig.MaxAttempts; attempt++ {
		result, err := c.processRequest(ctx, req)
		if err == nil {
			return result, nil
		}

		lastErr = err

		// Calculate delay with jitter
		delay := c.calculateDelay(attempt)
		if c.retryConfig.Jitter {
			delay = time.Duration(float64(delay) * (0.5 + rand.Float64()))
		}

		if attempt < c.retryConfig.MaxAttempts-1 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
				continue
			}
		}
	}

	return nil, fmt.Errorf("request failed after %d attempts: %w", c.retryConfig.MaxAttempts, lastErr)
}

func (c *ConsolidatedLocalAIClient) processRequest(ctx context.Context, req *types.GenerateRequest) (*types.GenerateResponse, error) {
	requestBody := map[string]interface{}{
		"model":             c.config.ModelName,
		"prompt":            req.Prompt,
		"temperature":       req.Temperature,
		"max_tokens":        req.MaxTokens,
		"top_p":             req.TopP,
		"frequency_penalty": req.FrequencyPenalty,
		"presence_penalty":  req.PresencePenalty,
	}

	if len(req.StopSequences) > 0 {
		requestBody["stop"] = req.StopSequences
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.getEndpoint()+"/v1/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.config.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var apiResponse struct {
		Choices []struct {
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(apiResponse.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from API")
	}

	return &types.GenerateResponse{
		Text:         apiResponse.Choices[0].Text,
		TokensUsed:   apiResponse.Usage.TotalTokens,
		FinishReason: apiResponse.Choices[0].FinishReason,
		Metadata: map[string]interface{}{
			"model":    c.config.ModelName,
			"endpoint": c.getEndpoint(),
		},
	}, nil
}

func (c *ConsolidatedLocalAIClient) handleFallback(ctx context.Context, req *types.GenerateRequest, reason string) (*types.GenerateResponse, error) {
	if !c.config.EnableFallback {
		return nil, fmt.Errorf("fallback not enabled")
	}

	// Simple fallback response
	return &types.GenerateResponse{
		Text:         "I'm currently experiencing technical difficulties. Please try again later.",
		TokensUsed:   0,
		FinishReason: "fallback",
		Metadata: map[string]interface{}{
			"fallback": true,
			"reason":   reason,
		},
	}, nil
}

func (c *ConsolidatedLocalAIClient) buildChatPrompt(messages []types.ChatMessage) string {
	var prompt strings.Builder
	for _, msg := range messages {
		prompt.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
	}
	return prompt.String()
}

func (c *ConsolidatedLocalAIClient) calculateDelay(attempt int) time.Duration {
	delay := time.Duration(float64(c.retryConfig.BaseDelay) * math.Pow(c.retryConfig.Multiplier, float64(attempt)))
	if delay > c.retryConfig.MaxDelay {
		delay = c.retryConfig.MaxDelay
	}
	return delay
}

// Placeholder implementations for layer-specific features
// These would be implemented in their respective layer4 modules

type ReasoningEngine struct{}

func NewReasoningEngine() *ReasoningEngine { return &ReasoningEngine{} }
func (r *ReasoningEngine) Close() error    { return nil }

type CollaborativeAI struct{}

func NewCollaborativeAI() *CollaborativeAI { return &CollaborativeAI{} }
func (c *CollaborativeAI) Close() error    { return nil }

type ModelRouter struct{}

func NewModelRouter() *ModelRouter  { return &ModelRouter{} }
func (m *ModelRouter) Close() error { return nil }

type PerformanceTracker struct{}

func NewPerformanceTracker() *PerformanceTracker { return &PerformanceTracker{} }
func (p *PerformanceTracker) Close() error       { return nil }

type EnhancedServer struct{}

func NewEnhancedServer(endpoints []string) *EnhancedServer { return &EnhancedServer{} }
func (e *EnhancedServer) Close() error                     { return nil }

type SemanticCaching struct{}

func NewSemanticCaching(cache *SemanticCache) *SemanticCaching { return &SemanticCaching{} }
func (s *SemanticCaching) Close() error                        { return nil }

type IntelligentRouting struct{}

func NewIntelligentRouting() *IntelligentRouting { return &IntelligentRouting{} }
func (i *IntelligentRouting) Close() error       { return nil }

type FallbackMechanism struct{}

func NewFallbackMechanism() *FallbackMechanism { return &FallbackMechanism{} }
func (f *FallbackMechanism) Close() error      { return nil }

// Circuit breaker methods
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if cb.state == StateClosed {
		return true
	}
	if cb.state == StateOpen {
		return time.Since(cb.lastFailureTime) > cb.timeout
	}
	return true // Half-open
}

func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.successCount++
	if cb.state == StateHalfOpen && cb.successCount >= cb.successThreshold {
		cb.state = StateClosed
		cb.failureCount = 0
		cb.successCount = 0
	}
}

func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount++
	cb.lastFailureTime = time.Now()

	if cb.failureCount >= cb.failureThreshold {
		cb.state = StateOpen
	}
}

func (cb *CircuitBreaker) GetState() string {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	switch cb.state {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half_open"
	default:
		return "unknown"
	}
}

func (cb *CircuitBreaker) GetFailureCount() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.failureCount
}

func (cb *CircuitBreaker) GetSuccessCount() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.successCount
}

// Cache methods
func (sc *SemanticCache) Get(prompt string) (*CacheEntry, bool) {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	entry, exists := sc.entries[prompt]
	if !exists {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.Timestamp) > sc.ttl {
		return nil, false
	}

	entry.AccessCount++
	return entry, true
}

func (sc *SemanticCache) Set(prompt, content string, tags []string) {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	// Simple LRU eviction
	if len(sc.entries) >= sc.maxSize {
		// Remove oldest entry
		var oldestKey string
		var oldestTime time.Time
		for key, entry := range sc.entries {
			if oldestTime.IsZero() || entry.Timestamp.Before(oldestTime) {
				oldestTime = entry.Timestamp
				oldestKey = key
			}
		}
		delete(sc.entries, oldestKey)
	}

	sc.entries[prompt] = &CacheEntry{
		Content:     content,
		Timestamp:   time.Now(),
		AccessCount: 1,
		Tags:        tags,
	}
}

func (sc *SemanticCache) Size() int {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return len(sc.entries)
}

func (sc *SemanticCache) GetHitRate() float64 {
	// Placeholder - would need proper hit/miss tracking
	return 0.0
}

// Load balancer methods
func (lb *ConsolidatedLoadBalancer) GetBestEndpoint() string {
	if len(lb.endpoints) == 0 {
		return ""
	}
	// Simple round-robin for now
	return lb.endpoints[0]
}

func (lb *ConsolidatedLoadBalancer) GetLoads() map[string]int {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	loads := make(map[string]int)
	for endpoint, load := range lb.loads {
		loads[endpoint] = load
	}
	return loads
}
