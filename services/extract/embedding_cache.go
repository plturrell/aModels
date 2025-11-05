package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

// EmbeddingCacheEntry represents a cached embedding entry
type EmbeddingCacheEntry struct {
	Embedding    []float32
	Relational   []float32 // For table embeddings that have both types
	Semantic     []float32 // For table embeddings that have both types
	Metadata     map[string]any
	CachedAt     time.Time
	ExpiresAt    time.Time
}

// EmbeddingCache provides caching for embeddings
type EmbeddingCache struct {
	mu          sync.RWMutex
	cache       map[string]*EmbeddingCacheEntry
	maxSize     int
	ttl         time.Duration
	cleanupTick *time.Ticker
	logger      *log.Logger
}

// NewEmbeddingCache creates a new embedding cache
func NewEmbeddingCache(maxSize int, ttl time.Duration, logger *log.Logger) *EmbeddingCache {
	cache := &EmbeddingCache{
		cache:   make(map[string]*EmbeddingCacheEntry),
		maxSize: maxSize,
		ttl:     ttl,
		logger:  logger,
	}

	// Start cleanup goroutine
	cache.cleanupTick = time.NewTicker(5 * time.Minute)
	go cache.cleanup()

	return cache
}

// cacheKey generates a cache key from artifact type and data
func (ec *EmbeddingCache) cacheKey(artifactType string, data interface{}) string {
	var keyData []byte
	switch d := data.(type) {
	case string:
		keyData = []byte(fmt.Sprintf("%s:%s", artifactType, d))
	case map[string]any:
		keyJSON, _ := json.Marshal(d)
		keyData = []byte(fmt.Sprintf("%s:%s", artifactType, string(keyJSON)))
	default:
		keyJSON, _ := json.Marshal(d)
		keyData = []byte(fmt.Sprintf("%s:%s", artifactType, string(keyJSON)))
	}

	hash := sha256.Sum256(keyData)
	return hex.EncodeToString(hash[:])
}

// Get retrieves an embedding from cache
func (ec *EmbeddingCache) Get(artifactType string, data interface{}) ([]float32, []float32, []float32, bool) {
	key := ec.cacheKey(artifactType, data)

	ec.mu.RLock()
	defer ec.mu.RUnlock()

	entry, exists := ec.cache[key]
	if !exists {
		return nil, nil, nil, false
	}

	// Check if expired
	if time.Now().After(entry.ExpiresAt) {
		return nil, nil, nil, false
	}

	// Return appropriate embedding(s)
	if entry.Relational != nil && entry.Semantic != nil {
		// Table embedding with both types
		return entry.Relational, entry.Semantic, nil, true
	} else if entry.Embedding != nil {
		// Single embedding
		return entry.Embedding, nil, nil, true
	}

	return nil, nil, nil, false
}

// Set stores an embedding in cache
func (ec *EmbeddingCache) Set(artifactType string, data interface{}, relational, semantic, embedding []float32, metadata map[string]any) {
	key := ec.cacheKey(artifactType, data)

	ec.mu.Lock()
	defer ec.mu.Unlock()

	// Evict oldest entries if cache is full
	if len(ec.cache) >= ec.maxSize {
		ec.evictOldest()
	}

	entry := &EmbeddingCacheEntry{
		Relational: relational,
		Semantic:   semantic,
		Embedding:  embedding,
		Metadata:   metadata,
		CachedAt:   time.Now(),
		ExpiresAt:  time.Now().Add(ec.ttl),
	}

	ec.cache[key] = entry
}

// evictOldest removes the oldest cache entry
func (ec *EmbeddingCache) evictOldest() {
	if len(ec.cache) == 0 {
		return
	}

	var oldestKey string
	var oldestTime time.Time
	first := true

	for key, entry := range ec.cache {
		if first || entry.CachedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.CachedAt
			first = false
		}
	}

	if oldestKey != "" {
		delete(ec.cache, oldestKey)
	}
}

// cleanup removes expired entries periodically
func (ec *EmbeddingCache) cleanup() {
	for range ec.cleanupTick.C {
		ec.mu.Lock()
		now := time.Now()
		for key, entry := range ec.cache {
			if now.After(entry.ExpiresAt) {
				delete(ec.cache, key)
			}
		}
		ec.mu.Unlock()
	}
}

// Clear clears all cache entries
func (ec *EmbeddingCache) Clear() {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.cache = make(map[string]*EmbeddingCacheEntry)
}

// Stats returns cache statistics
func (ec *EmbeddingCache) Stats() map[string]interface{} {
	ec.mu.RLock()
	defer ec.mu.RUnlock()

	return map[string]interface{}{
		"size":      len(ec.cache),
		"max_size":  ec.maxSize,
		"ttl":       ec.ttl.String(),
	}
}

