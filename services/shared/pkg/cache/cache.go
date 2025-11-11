package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// Cache provides a multi-level cache with in-memory (LRU) and Redis backends.
type Cache struct {
	memory    *memoryCache
	redis     *redis.Client
	redisURL  string
	enabled   bool
	mu        sync.RWMutex
	stats     CacheStats
}

// CacheStats tracks cache performance metrics.
type CacheStats struct {
	MemoryHits   int64
	MemoryMisses int64
	RedisHits    int64
	RedisMisses  int64
	MemorySize   int64
}

// Config configures the cache.
type Config struct {
	MemorySize int           // Maximum number of items in memory cache (default: 1000)
	RedisURL   string        // Redis connection URL (optional)
	DefaultTTL time.Duration // Default TTL for cached items (default: 5 minutes)
	Enabled    bool          // Enable/disable caching (default: true)
}

// DefaultConfig returns default cache configuration.
func DefaultConfig() Config {
	return Config{
		MemorySize: 1000,
		DefaultTTL: 5 * time.Minute,
		Enabled:    true,
	}
}

// NewMultiLevelCache creates a new multi-level cache.
func NewMultiLevelCache(config Config) (*Cache, error) {
	if !config.Enabled {
		return &Cache{enabled: false}, nil
	}

	cache := &Cache{
		memory:   newMemoryCache(config.MemorySize),
		redisURL: config.RedisURL,
		enabled:  true,
	}

	// Initialize Redis if URL provided
	if config.RedisURL != "" {
		opt, err := redis.ParseURL(config.RedisURL)
		if err != nil {
			return nil, fmt.Errorf("parse redis url: %w", err)
		}
		cache.redis = redis.NewClient(opt)
		
		// Test connection
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := cache.redis.Ping(ctx).Err(); err != nil {
			// Redis unavailable, continue with memory-only cache
			cache.redis = nil
		}
	}

	return cache, nil
}

// Get retrieves a value from cache.
func (c *Cache) Get(ctx context.Context, key string) ([]byte, error) {
	if !c.enabled {
		return nil, nil
	}

	// Try memory cache first
	if val := c.memory.get(key); val != nil {
		c.mu.Lock()
		c.stats.MemoryHits++
		c.mu.Unlock()
		return val, nil
	}
	c.mu.Lock()
	c.stats.MemoryMisses++
	c.mu.Unlock()

	// Try Redis cache
	if c.redis != nil {
		val, err := c.redis.Get(ctx, key).Bytes()
		if err == nil {
			// Store in memory cache for faster access
			c.memory.set(key, val)
			c.mu.Lock()
			c.stats.RedisHits++
			c.mu.Unlock()
			return val, nil
		}
		if err != redis.Nil {
			// Redis error, but not a cache miss
			return nil, err
		}
		c.mu.Lock()
		c.stats.RedisMisses++
		c.mu.Unlock()
	}

	return nil, nil // Cache miss
}

// Set stores a value in cache.
func (c *Cache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	if !c.enabled {
		return nil
	}

	if ttl <= 0 {
		ttl = 5 * time.Minute
	}

	// Store in memory cache
	c.memory.set(key, value)

	// Store in Redis if available
	if c.redis != nil {
		if err := c.redis.Set(ctx, key, value, ttl).Err(); err != nil {
			// Redis error, but continue with memory cache
		}
	}

	return nil
}

// Delete removes a value from cache.
func (c *Cache) Delete(ctx context.Context, key string) error {
	if !c.enabled {
		return nil
	}

	c.memory.delete(key)

	if c.redis != nil {
		return c.redis.Del(ctx, key).Err()
	}

	return nil
}

// GetJSON retrieves and unmarshals a JSON value from cache.
func (c *Cache) GetJSON(ctx context.Context, key string, dest interface{}) error {
	data, err := c.Get(ctx, key)
	if err != nil {
		return err
	}
	if data == nil {
		return nil // Cache miss
	}
	return json.Unmarshal(data, dest)
}

// SetJSON marshals and stores a JSON value in cache.
func (c *Cache) SetJSON(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return c.Set(ctx, key, data, ttl)
}

// Stats returns cache performance statistics.
func (c *Cache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return CacheStats{
		MemoryHits:   c.stats.MemoryHits,
		MemoryMisses: c.stats.MemoryMisses,
		RedisHits:    c.stats.RedisHits,
		RedisMisses:  c.stats.RedisMisses,
		MemorySize:   int64(c.memory.size()),
	}
}

// Close closes the cache and releases resources.
func (c *Cache) Close() error {
	if c.redis != nil {
		return c.redis.Close()
	}
	return nil
}

// memoryCache provides a simple in-memory LRU cache.
type memoryCache struct {
	items map[string]*cacheItem
	mu    sync.RWMutex
	maxSize int
}

type cacheItem struct {
	value     []byte
	timestamp time.Time
}

func newMemoryCache(maxSize int) *memoryCache {
	if maxSize <= 0 {
		maxSize = 1000
	}
	return &memoryCache{
		items:   make(map[string]*cacheItem),
		maxSize: maxSize,
	}
}

func (mc *memoryCache) get(key string) []byte {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	item, ok := mc.items[key]
	if !ok {
		return nil
	}
	// Simple LRU: update timestamp on access
	item.timestamp = time.Now()
	return item.value
}

func (mc *memoryCache) set(key string, value []byte) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	// Evict oldest if at capacity
	if len(mc.items) >= mc.maxSize && mc.items[key] == nil {
		mc.evictOldest()
	}

	mc.items[key] = &cacheItem{
		value:     value,
		timestamp: time.Now(),
	}
}

func (mc *memoryCache) delete(key string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	delete(mc.items, key)
}

func (mc *memoryCache) size() int {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	return len(mc.items)
}

func (mc *memoryCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time
	first := true

	for key, item := range mc.items {
		if first || item.timestamp.Before(oldestTime) {
			oldestKey = key
			oldestTime = item.timestamp
			first = false
		}
	}

	if oldestKey != "" {
		delete(mc.items, oldestKey)
	}
}

