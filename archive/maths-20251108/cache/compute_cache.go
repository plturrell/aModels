package cache

import (
	"crypto/sha256"
	"encoding/binary"
	"sync"
	"time"
)

// ComputeCache provides intelligent caching for repeated computations
// Features: LRU eviction, hit rate monitoring, thread-safe operations
type ComputeCache struct {
	mu         sync.RWMutex
	cache      map[string]interface{}
	maxSize    int
	hits       uint64
	misses     uint64
	lruList    []string
	accessTime map[string]time.Time
}

// CacheEntry represents a cached computation result
type CacheEntry struct {
	Value       interface{}
	Timestamp   time.Time
	AccessCount uint64
}

// NewComputeCache creates a new compute cache with specified maximum size
func NewComputeCache(maxSize int) *ComputeCache {
	return &ComputeCache{
		cache:      make(map[string]interface{}),
		maxSize:    maxSize,
		lruList:    make([]string, 0, maxSize),
		accessTime: make(map[string]time.Time),
	}
}

// Get retrieves a value from the cache
func (c *ComputeCache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	val, ok := c.cache[key]
	c.mu.RUnlock()

	if ok {
		c.mu.Lock()
		c.hits++
		c.accessTime[key] = time.Now()
		c.mu.Unlock()
		return val, true
	}

	c.mu.Lock()
	c.misses++
	c.mu.Unlock()
	return nil, false
}

// Put stores a value in the cache with LRU eviction
func (c *ComputeCache) Put(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if key already exists
	if _, exists := c.cache[key]; exists {
		c.cache[key] = value
		c.accessTime[key] = time.Now()
		return
	}

	// Evict oldest entry if cache is full
	if len(c.cache) >= c.maxSize {
		c.evictOldest()
	}

	c.cache[key] = value
	c.accessTime[key] = time.Now()
	c.lruList = append(c.lruList, key)
}

// evictOldest removes the least recently used entry
func (c *ComputeCache) evictOldest() {
	if len(c.lruList) == 0 {
		return
	}

	// Find oldest entry by access time
	var oldestKey string
	var oldestTime time.Time
	first := true

	for key, accessTime := range c.accessTime {
		if first || accessTime.Before(oldestTime) {
			oldestKey = key
			oldestTime = accessTime
			first = false
		}
	}

	if oldestKey != "" {
		delete(c.cache, oldestKey)
		delete(c.accessTime, oldestKey)

		// Remove from LRU list
		for i, key := range c.lruList {
			if key == oldestKey {
				c.lruList = append(c.lruList[:i], c.lruList[i+1:]...)
				break
			}
		}
	}
}

// HitRate returns the cache hit rate as a percentage
func (c *ComputeCache) HitRate() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.misses
	if total == 0 {
		return 0
	}
	return float64(c.hits) / float64(total) * 100
}

// Stats returns cache statistics
func (c *ComputeCache) Stats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return map[string]interface{}{
		"hits":        c.hits,
		"misses":      c.misses,
		"hit_rate":    c.HitRate(),
		"size":        len(c.cache),
		"max_size":    c.maxSize,
		"utilization": float64(len(c.cache)) / float64(c.maxSize) * 100,
	}
}

// Clear removes all entries from the cache
func (c *ComputeCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache = make(map[string]interface{})
	c.lruList = make([]string, 0, c.maxSize)
	c.accessTime = make(map[string]time.Time)
	c.hits = 0
	c.misses = 0
}

// Size returns the current number of cached entries
func (c *ComputeCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.cache)
}

// VectorCacheKey generates a cache key from two vectors
func VectorCacheKey(v1, v2 []float64) string {
	h := sha256.New()

	// Write vector lengths first for uniqueness
	binary.Write(h, binary.LittleEndian, uint64(len(v1)))
	binary.Write(h, binary.LittleEndian, uint64(len(v2)))

	// Write vector data
	for _, v := range v1 {
		binary.Write(h, binary.LittleEndian, v)
	}
	for _, v := range v2 {
		binary.Write(h, binary.LittleEndian, v)
	}

	return string(h.Sum(nil))
}

// MatrixCacheKey generates a cache key from matrix dimensions and data
func MatrixCacheKey(rows, cols int, data []float64) string {
	h := sha256.New()

	binary.Write(h, binary.LittleEndian, uint64(rows))
	binary.Write(h, binary.LittleEndian, uint64(cols))

	for _, v := range data {
		binary.Write(h, binary.LittleEndian, v)
	}

	return string(h.Sum(nil))
}

// VectorCacheKeyM wraps VectorCacheKey as a method for compatibility with callers.
func (c *ComputeCache) VectorCacheKey(v1, v2 []float64) string {
	return VectorCacheKey(v1, v2)
}

// OperationCacheKeyM wraps OperationCacheKey as a method.
func (c *ComputeCache) OperationCacheKey(op string, params ...interface{}) string {
	return OperationCacheKey(op, params...)
}

// OperationCacheKey generates a cache key for operations with parameters
func OperationCacheKey(op string, params ...interface{}) string {
	h := sha256.New()

	h.Write([]byte(op))

	for _, param := range params {
		switch v := param.(type) {
		case int:
			binary.Write(h, binary.LittleEndian, int64(v))
		case float64:
			binary.Write(h, binary.LittleEndian, v)
		case string:
			h.Write([]byte(v))
		case []float64:
			for _, val := range v {
				binary.Write(h, binary.LittleEndian, val)
			}
		}
	}

	return string(h.Sum(nil))
}

// Global cache instance for shared use across the math library
var globalCache *ComputeCache
var cacheOnce sync.Once

// GetGlobalCache returns the global cache instance
func GetGlobalCache() *ComputeCache {
	cacheOnce.Do(func() {
		globalCache = NewComputeCache(10000) // Default 10k entries
	})
	return globalCache
}

// SetGlobalCacheSize sets the size of the global cache
func SetGlobalCacheSize(size int) {
	cacheOnce.Do(func() {
		globalCache = NewComputeCache(size)
	})
}

// CachedCosine computes cosine similarity with caching
func CachedCosine(v1, v2 []float64) float64 {
	cache := GetGlobalCache()
	key := VectorCacheKey(v1, v2)

	if result, ok := cache.Get(key); ok {
		return result.(float64)
	}

	// Compute similarity (this would call the actual math function)
	// For now, return 0 - in real implementation, call maths.CosineAuto
	result := 0.0 // maths.CosineAuto(v1, v2)

	cache.Put(key, result)
	return result
}

// CachedDot computes dot product with caching
func CachedDot(v1, v2 []float64) float64 {
	cache := GetGlobalCache()
	key := VectorCacheKey(v1, v2)

	if result, ok := cache.Get(key); ok {
		return result.(float64)
	}

	// Compute dot product
	result := 0.0 // maths.DotAuto(v1, v2)

	cache.Put(key, result)
	return result
}

// CachedMatMul computes matrix multiplication with caching
func CachedMatMul(a, b [][]float64) [][]float64 {
	cache := GetGlobalCache()

	// Flatten matrices for key generation
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])

	flatA := make([]float64, rowsA*colsA)
	flatB := make([]float64, rowsB*colsB)

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			flatA[i*colsA+j] = a[i][j]
		}
	}

	for i := 0; i < rowsB; i++ {
		for j := 0; j < colsB; j++ {
			flatB[i*colsB+j] = b[i][j]
		}
	}

	key := OperationCacheKey("matmul", rowsA, colsA, rowsB, colsB, flatA, flatB)

	if result, ok := cache.Get(key); ok {
		return result.([][]float64)
	}

	// Compute matrix multiplication
	result := make([][]float64, rowsA)
	for i := range result {
		result[i] = make([]float64, colsB)
	}

	// Actual computation would go here
	// maths.MatMul(a, b, result)

	cache.Put(key, result)
	return result
}
