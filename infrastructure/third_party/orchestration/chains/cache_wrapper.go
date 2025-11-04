// Copyright 2025 AgenticAI ETH Contributors
// SPDX-License-Identifier: MIT

package chains

import (
	"context"
	"os"
	"strconv"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/cache"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/cache/inmemory"
)

// CacheConfig holds configuration for chain-level caching
type CacheConfig struct {
	// Enabled controls whether caching is active
	Enabled bool

	// TTL is the time-to-live for cached entries
	TTL string

	// MaxSize is the maximum number of entries in the cache
	MaxSize int

	// SimilarityThreshold is the minimum similarity score for semantic cache hits
	SimilarityThreshold float64
}

// DefaultCacheConfig returns a default cache configuration
func DefaultCacheConfig() *CacheConfig {
	enabled := false
	if val := os.Getenv("CACHE_SEMANTIC"); val != "" {
		if parsed, err := strconv.ParseBool(val); err == nil {
			enabled = parsed
		}
	}

	return &CacheConfig{
		Enabled:             enabled,
		TTL:                 "24h",
		MaxSize:             1000,
		SimilarityThreshold: 0.8,
	}
}

// WithCaching wraps an LLM with caching capabilities
func WithCaching(llm llms.Model, config *CacheConfig) (llms.Model, error) {
	if config == nil {
		config = DefaultCacheConfig()
	}

	if !config.Enabled {
		return llm, nil
	}

	// Create in-memory cache backend
	ctx := context.Background()
	backend, err := inmemory.New(ctx, inmemory.WithTTL(config.TTL), inmemory.WithMaxSize(config.MaxSize))
	if err != nil {
		return nil, err
	}

	// Wrap the LLM with caching
	return cache.New(llm, backend), nil
}

// WithCachingOption is a chain option that enables caching
func WithCachingOption(config *CacheConfig) ChainCallOption {
	return func(o *chainCallOption) {
		// This will be used when creating the chain to wrap the LLM
		o.CacheConfig = config
	}
}

// CacheProvider interface for different cache backends
type CacheProvider interface {
	Get(ctx context.Context, key string) *llms.ContentResponse
	Put(ctx context.Context, key string, response *llms.ContentResponse)
	FindSimilar(ctx context.Context, prompt string, threshold float64) []*llms.ContentResponse
}

// SemanticCacheClient implements CacheProvider using HANA semantic cache
type SemanticCacheClient struct {
	// This would connect to the LocalAI server's semantic cache
	// For now, we'll use the in-memory cache as a placeholder
	backend cache.Backend
}

// NewSemanticCacheClient creates a new semantic cache client
func NewSemanticCacheClient(backend cache.Backend) *SemanticCacheClient {
	return &SemanticCacheClient{
		backend: backend,
	}
}

// Get retrieves a cached response
func (c *SemanticCacheClient) Get(ctx context.Context, key string) *llms.ContentResponse {
	return c.backend.Get(ctx, key)
}

// Put stores a response in cache
func (c *SemanticCacheClient) Put(ctx context.Context, key string, response *llms.ContentResponse) {
	c.backend.Put(ctx, key, response)
}

// FindSimilar finds semantically similar cached responses
func (c *SemanticCacheClient) FindSimilar(ctx context.Context, prompt string, threshold float64) []*llms.ContentResponse {
	// This would implement semantic similarity search
	// For now, return empty slice
	return []*llms.ContentResponse{}
}
