package search

import (
	"encoding/json"
	"sync"
	"time"
)

type cacheEntry struct {
	value   []byte
	expires time.Time
}

type jsonCache struct {
	mu         sync.RWMutex
	entries    map[string]cacheEntry
	defaultTTL time.Duration
}

func newJSONCache(ttl time.Duration) *jsonCache {
	if ttl <= 0 {
		ttl = 5 * time.Minute
	}
	return &jsonCache{
		entries:    make(map[string]cacheEntry),
		defaultTTL: ttl,
	}
}

func (c *jsonCache) GetJSON(key string, dest interface{}) bool {
	if c == nil {
		return false
	}

	c.mu.RLock()
	entry, ok := c.entries[key]
	c.mu.RUnlock()
	if !ok || time.Now().After(entry.expires) {
		if ok {
			c.mu.Lock()
			delete(c.entries, key)
			c.mu.Unlock()
		}
		return false
	}

	if err := json.Unmarshal(entry.value, dest); err != nil {
		return false
	}
	return true
}

func (c *jsonCache) SetJSON(key string, value interface{}, ttl time.Duration) {
	if c == nil {
		return
	}

	if ttl <= 0 {
		ttl = c.defaultTTL
	}

	payload, err := json.Marshal(value)
	if err != nil {
		return
	}

	c.mu.Lock()
	c.entries[key] = cacheEntry{
		value:   payload,
		expires: time.Now().Add(ttl),
	}
	c.mu.Unlock()
}
