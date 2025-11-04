// Copyright 2025 AgenticAI ETH Contributors
// SPDX-License-Identifier: MIT

// Package cache provides semantic caching middleware for orchestration chains
package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

// CacheProvider defines the interface for cache operations
type CacheProvider interface {
	// Get retrieves a cached response for the given key
	Get(ctx context.Context, key string) (*CachedResponse, error)

	// Set stores a response in the cache
	Set(ctx context.Context, key string, response *CachedResponse, ttl time.Duration) error

	// FindSemanticSimilar finds semantically similar cached responses
	FindSemanticSimilar(ctx context.Context, prompt, model, domain string, threshold float64, limit int) ([]*CachedResponse, error)
}

// CachedResponse represents a cached LLM response
type CachedResponse struct {
	Content     string            `json:"content"`
	Model       string            `json:"model"`
	Domain      string            `json:"domain"`
	TokensUsed  int               `json:"tokens_used"`
	Temperature float64           `json:"temperature"`
	MaxTokens   int               `json:"max_tokens"`
	CreatedAt   time.Time         `json:"created_at"`
	Metadata    map[string]string `json:"metadata"`
}

// CacheConfig holds configuration for the cache middleware
type CacheConfig struct {
	// Enabled controls whether caching is active
	Enabled bool

	// DefaultTTL is the default time-to-live for cached responses
	DefaultTTL time.Duration

	// SimilarityThreshold is the minimum similarity score for semantic cache hits
	SimilarityThreshold float64

	// MaxSemanticResults is the maximum number of semantic results to return
	MaxSemanticResults int

	// CacheTags are additional tags to include with cached responses
	CacheTags []string
}

// DefaultCacheConfig returns a default cache configuration
func DefaultCacheConfig() *CacheConfig {
	enabled := true
	if val := os.Getenv("CACHE_SEMANTIC"); val != "" {
		if b, err := strconv.ParseBool(val); err == nil {
			enabled = b
		}
	}

	return &CacheConfig{
		Enabled:             enabled,
		DefaultTTL:          24 * time.Hour,
		SimilarityThreshold: 0.8,
		MaxSemanticResults:  5,
		CacheTags:           []string{"orchestration", "chain"},
	}
}

// CacheMiddleware wraps an LLM with caching capabilities
type CacheMiddleware struct {
	llm    llms.Model
	cache  CacheProvider
	config *CacheConfig
}

// NewCacheMiddleware creates a new cache middleware
func NewCacheMiddleware(llm llms.Model, cache CacheProvider, config *CacheConfig) *CacheMiddleware {
	if config == nil {
		config = DefaultCacheConfig()
	}

	return &CacheMiddleware{
		llm:    llm,
		cache:  cache,
		config: config,
	}
}

// Call implements the llms.Model interface with caching support.
func (cm *CacheMiddleware) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	if !cm.config.Enabled {
		return cm.llm.Call(ctx, prompt, options...)
	}

	// Generate cache key
	cacheKey := cm.generateCacheKey(prompt, options...)

	// Try to get from cache first
	if cached, err := cm.cache.Get(ctx, cacheKey); err == nil && cached != nil {
		return cached.Content, nil
	}

	// Try semantic cache
	if cm.config.SimilarityThreshold > 0 {
		if similar, err := cm.cache.FindSemanticSimilar(ctx, prompt, "auto", "general", cm.config.SimilarityThreshold, cm.config.MaxSemanticResults); err == nil && len(similar) > 0 {
			// Use the most similar response
			best := similar[0]
			return best.Content, nil
		}
	}

	// Generate new response
	response, err := cm.llm.Call(ctx, prompt, options...)
	if err != nil {
		return "", err
	}

	// Cache the response
	cachedResp := &CachedResponse{
		Content:     response,
		Model:       "auto",
		Domain:      "general",
		TokensUsed:  len(strings.Split(response, " ")),
		Temperature: 0.7,  // Default
		MaxTokens:   2048, // Default
		CreatedAt:   time.Now(),
		Metadata: map[string]string{
			"source": "orchestration_chain",
			"tags":   strings.Join(cm.config.CacheTags, ","),
		},
	}

	// Store in cache asynchronously
	go func() {
		if err := cm.cache.Set(context.Background(), cacheKey, cachedResp, cm.config.DefaultTTL); err != nil {
			// Log error but don't fail the request
			fmt.Printf("⚠️ Failed to cache response: %v\n", err)
		}
	}()

	return response, nil
}

// GenerateContent implements the llms.Model interface with caching
func (cm *CacheMiddleware) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	if !cm.config.Enabled {
		return cm.llm.GenerateContent(ctx, messages, options...)
	}

	// Convert messages to prompt string for caching
	var prompt strings.Builder
	for _, msg := range messages {
		prompt.WriteString(string(msg.Role))
		prompt.WriteString(": ")
		for _, part := range msg.Parts {
			if textPart, ok := part.(llms.TextContent); ok {
				prompt.WriteString(textPart.Text)
			}
		}
		prompt.WriteString("\n")
	}

	// Generate cache key
	cacheKey := cm.generateCacheKey(prompt.String(), options...)

	// Try to get from cache first
	if cached, err := cm.cache.Get(ctx, cacheKey); err == nil && cached != nil {
		// Convert cached response back to ContentResponse format
		return &llms.ContentResponse{
			Choices: []*llms.ContentChoice{
				{
					Content: cached.Content,
					GenerationInfo: map[string]any{
						"model":       cached.Model,
						"tokens_used": cached.TokensUsed,
						"cached":      true,
					},
				},
			},
		}, nil
	}

	// Generate new response
	response, err := cm.llm.GenerateContent(ctx, messages, options...)
	if err != nil {
		return nil, err
	}

	// Extract content for caching
	var content string
	if len(response.Choices) > 0 {
		content = response.Choices[0].Content
	}

	// Cache the response
	cachedResp := &CachedResponse{
		Content:     content,
		Model:       "auto",
		Domain:      "general",
		TokensUsed:  len(strings.Split(content, " ")),
		Temperature: 0.7,  // Default
		MaxTokens:   2048, // Default
		CreatedAt:   time.Now(),
		Metadata: map[string]string{
			"source": "orchestration_chain",
			"tags":   strings.Join(cm.config.CacheTags, ","),
		},
	}

	// Store in cache asynchronously
	go func() {
		if err := cm.cache.Set(context.Background(), cacheKey, cachedResp, cm.config.DefaultTTL); err != nil {
			// Log error but don't fail the request
			fmt.Printf("⚠️ Failed to cache response: %v\n", err)
		}
	}()

	return response, nil
}

// generateCacheKey creates a cache key for the given prompt and options
func (cm *CacheMiddleware) generateCacheKey(prompt string, options ...llms.CallOption) string {
	// Create a hash of the prompt and options
	hasher := sha256.New()
	hasher.Write([]byte(prompt))

	// Include options in the hash
	opts := &llms.CallOptions{}
	for _, opt := range options {
		opt(opts)
	}

	// Add temperature and max tokens to the key
	hasher.Write([]byte(fmt.Sprintf("temp:%.2f", opts.Temperature)))
	hasher.Write([]byte(fmt.Sprintf("max:%d", opts.MaxTokens)))

	return hex.EncodeToString(hasher.Sum(nil))
}

// GetInputKeys returns the input keys (delegates to wrapped LLM)
func (cm *CacheMiddleware) GetInputKeys() []string {
	if inputKeyer, ok := cm.llm.(interface{ GetInputKeys() []string }); ok {
		return inputKeyer.GetInputKeys()
	}
	return []string{}
}

// GetOutputKeys returns the output keys (delegates to wrapped LLM)
func (cm *CacheMiddleware) GetOutputKeys() []string {
	if outputKeyer, ok := cm.llm.(interface{ GetOutputKeys() []string }); ok {
		return outputKeyer.GetOutputKeys()
	}
	return []string{}
}

// GetMemory returns the memory (delegates to wrapped LLM)
func (cm *CacheMiddleware) GetMemory() interface{} {
	if memoryHaver, ok := cm.llm.(interface{ GetMemory() interface{} }); ok {
		return memoryHaver.GetMemory()
	}
	return nil
}

// GetCallbackHandler returns the callback handler (delegates to wrapped LLM)
func (cm *CacheMiddleware) GetCallbackHandler() interface{} {
	if callbackHaver, ok := cm.llm.(interface{ GetCallbackHandler() interface{} }); ok {
		return callbackHaver.GetCallbackHandler()
	}
	return nil
}
