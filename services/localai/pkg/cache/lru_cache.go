package cache

import (
	"container/list"
	"context"
	"sync"
	"time"
)

// LRUCache implements an LRU (Least Recently Used) cache with TTL support
type LRUCache struct {
	capacity   int
	items      map[string]*list.Element
	evictList  *list.List
	mu         sync.RWMutex
	stats      *CacheStats
	onEvict    func(key string, value interface{})
	defaultTTL time.Duration
}

// CacheEntry represents a cached item with metadata
type CacheEntry struct {
	Key       string
	Value     interface{}
	ExpiresAt time.Time
	Size      int64
	CreatedAt time.Time
	AccessAt  time.Time
	HitCount  int64
}

// CacheStats tracks cache performance metrics
type CacheStats struct {
	Hits          int64
	Misses        int64
	Evictions     int64
	Expirations   int64
	Entries       int64
	TotalSize     int64
	AvgAccessTime time.Duration
	mu            sync.RWMutex
}

// LRUConfig holds configuration for LRU cache
type LRUConfig struct {
	Capacity   int
	DefaultTTL time.Duration
	OnEvict    func(key string, value interface{})
}

// NewLRUCache creates a new LRU cache
func NewLRUCache(cfg *LRUConfig) *LRUCache {
	if cfg == nil {
		cfg = &LRUConfig{
			Capacity:   1000,
			DefaultTTL: 1 * time.Hour,
		}
	}

	return &LRUCache{
		capacity:   cfg.Capacity,
		items:      make(map[string]*list.Element),
		evictList:  list.New(),
		stats:      &CacheStats{},
		onEvict:    cfg.OnEvict,
		defaultTTL: cfg.DefaultTTL,
	}
}

// Get retrieves a value from the cache
func (c *LRUCache) Get(ctx context.Context, key string) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	start := time.Now()
	defer func() {
		c.stats.mu.Lock()
		c.stats.AvgAccessTime = (c.stats.AvgAccessTime + time.Since(start)) / 2
		c.stats.mu.Unlock()
	}()

	element, found := c.items[key]
	if !found {
		c.stats.mu.Lock()
		c.stats.Misses++
		c.stats.mu.Unlock()
		return nil, false
	}

	entry := element.Value.(*CacheEntry)

	// Check if expired
	if time.Now().After(entry.ExpiresAt) {
		c.removeElement(element)
		c.stats.mu.Lock()
		c.stats.Misses++
		c.stats.Expirations++
		c.stats.mu.Unlock()
		return nil, false
	}

	// Update access time and hit count
	entry.AccessAt = time.Now()
	entry.HitCount++

	// Move to front (most recently used)
	c.evictList.MoveToFront(element)

	c.stats.mu.Lock()
	c.stats.Hits++
	c.stats.mu.Unlock()

	return entry.Value, true
}

// Set adds or updates a value in the cache
func (c *LRUCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if ttl == 0 {
		ttl = c.defaultTTL
	}

	now := time.Now()
	entry := &CacheEntry{
		Key:       key,
		Value:     value,
		ExpiresAt: now.Add(ttl),
		CreatedAt: now,
		AccessAt:  now,
		HitCount:  0,
		Size:      estimateSize(value),
	}

	// Check if key already exists
	if element, found := c.items[key]; found {
		// Update existing entry
		oldEntry := element.Value.(*CacheEntry)
		c.stats.mu.Lock()
		c.stats.TotalSize -= oldEntry.Size
		c.stats.TotalSize += entry.Size
		c.stats.mu.Unlock()

		element.Value = entry
		c.evictList.MoveToFront(element)
		return nil
	}

	// Add new entry
	element := c.evictList.PushFront(entry)
	c.items[key] = element

	c.stats.mu.Lock()
	c.stats.Entries++
	c.stats.TotalSize += entry.Size
	c.stats.mu.Unlock()

	// Evict if over capacity
	if c.evictList.Len() > c.capacity {
		c.evict()
	}

	return nil
}

// Delete removes a key from the cache
func (c *LRUCache) Delete(ctx context.Context, key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	element, found := c.items[key]
	if !found {
		return false
	}

	c.removeElement(element)
	return true
}

// Clear removes all entries from the cache
func (c *LRUCache) Clear(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.items = make(map[string]*list.Element)
	c.evictList.Init()

	c.stats.mu.Lock()
	c.stats.Entries = 0
	c.stats.TotalSize = 0
	c.stats.mu.Unlock()
}

// Len returns the number of entries in the cache
func (c *LRUCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.evictList.Len()
}

// GetStats returns cache statistics
func (c *LRUCache) GetStats() map[string]interface{} {
	c.stats.mu.RLock()
	defer c.stats.mu.RUnlock()

	total := c.stats.Hits + c.stats.Misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(c.stats.Hits) / float64(total)
	}

	return map[string]interface{}{
		"hits":            c.stats.Hits,
		"misses":          c.stats.Misses,
		"hit_rate":        hitRate,
		"evictions":       c.stats.Evictions,
		"expirations":     c.stats.Expirations,
		"entries":         c.stats.Entries,
		"capacity":        c.capacity,
		"total_size_mb":   float64(c.stats.TotalSize) / 1024 / 1024,
		"avg_access_time": c.stats.AvgAccessTime.Microseconds(),
	}
}

// CleanupExpired removes expired entries
func (c *LRUCache) CleanupExpired(ctx context.Context) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	expired := 0

	for element := c.evictList.Back(); element != nil; {
		entry := element.Value.(*CacheEntry)
		prev := element.Prev()

		if now.After(entry.ExpiresAt) {
			c.removeElement(element)
			expired++
			c.stats.mu.Lock()
			c.stats.Expirations++
			c.stats.mu.Unlock()
		}

		element = prev
	}

	return expired
}

// StartCleanupTimer starts a background goroutine to clean up expired entries
func (c *LRUCache) StartCleanupTimer(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				count := c.CleanupExpired(ctx)
				if count > 0 {
					// Log cleanup if needed
				}
			}
		}
	}()
}

// evict removes the least recently used item
func (c *LRUCache) evict() {
	element := c.evictList.Back()
	if element != nil {
		c.removeElement(element)
		c.stats.mu.Lock()
		c.stats.Evictions++
		c.stats.mu.Unlock()
	}
}

// removeElement removes an element from the cache
func (c *LRUCache) removeElement(element *list.Element) {
	c.evictList.Remove(element)
	entry := element.Value.(*CacheEntry)
	delete(c.items, entry.Key)

	c.stats.mu.Lock()
	c.stats.Entries--
	c.stats.TotalSize -= entry.Size
	c.stats.mu.Unlock()

	if c.onEvict != nil {
		c.onEvict(entry.Key, entry.Value)
	}
}

// estimateSize estimates the memory size of a value
func estimateSize(value interface{}) int64 {
	switch v := value.(type) {
	case string:
		return int64(len(v))
	case []byte:
		return int64(len(v))
	case int, int32, int64, uint, uint32, uint64, float32, float64, bool:
		return 8
	default:
		// Default estimate for complex types
		return 1024
	}
}
