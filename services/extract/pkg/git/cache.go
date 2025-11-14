package git

import (
	"sync"
	"time"
)

// CacheEntry represents a cached response with expiration
type CacheEntry struct {
	Data      interface{}
	ExpiresAt time.Time
}

// ResponseCache provides simple in-memory caching for GET requests
type ResponseCache struct {
	mu    sync.RWMutex
	items map[string]*CacheEntry
	ttl   time.Duration
}

// NewResponseCache creates a new response cache with the specified TTL
func NewResponseCache(ttl time.Duration) *ResponseCache {
	cache := &ResponseCache{
		items: make(map[string]*CacheEntry),
		ttl:   ttl,
	}
	
	// Start cleanup goroutine
	go cache.cleanup()
	
	return cache
}

// Get retrieves a cached value if it exists and hasn't expired
func (c *ResponseCache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	entry, exists := c.items[key]
	if !exists {
		return nil, false
	}
	
	if time.Now().After(entry.ExpiresAt) {
		return nil, false
	}
	
	return entry.Data, true
}

// Set stores a value in the cache with TTL
func (c *ResponseCache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.items[key] = &CacheEntry{
		Data:      value,
		ExpiresAt: time.Now().Add(c.ttl),
	}
}

// Delete removes a key from the cache
func (c *ResponseCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, key)
}

// Clear removes all entries from the cache
func (c *ResponseCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]*CacheEntry)
}

// cleanup periodically removes expired entries
func (c *ResponseCache) cleanup() {
	ticker := time.NewTicker(c.ttl)
	defer ticker.Stop()
	
	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		for key, entry := range c.items {
			if now.After(entry.ExpiresAt) {
				delete(c.items, key)
			}
		}
		c.mu.Unlock()
	}
}

