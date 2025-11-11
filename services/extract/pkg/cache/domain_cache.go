package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// DomainCache provides caching for domain configurations from LocalAI
type DomainCache struct {
	redisClient *redis.Client
	inMemory    map[string]cachedDomain
	mu          sync.RWMutex
	ttl         time.Duration
	logger      *log.Logger
}

type cachedDomain struct {
	Config    map[string]interface{}
	ExpiresAt time.Time
}

// NewDomainCache creates a new domain cache
func NewDomainCache(redisClient *redis.Client, logger *log.Logger) *DomainCache {
	ttl := 1 * time.Hour // Default TTL
	if val := os.Getenv("DOMAIN_CACHE_TTL_MINUTES"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			ttl = time.Duration(parsed) * time.Minute
		}
	}

	return &DomainCache{
		redisClient: redisClient,
		inMemory:    make(map[string]cachedDomain),
		ttl:         ttl,
		logger:      logger,
	}
}

// Get retrieves domain configurations from cache
func (dc *DomainCache) Get(ctx context.Context) (map[string]interface{}, error) {
	cacheKey := "domain:configs:all"

	// Try Redis first if available
	if dc.redisClient != nil {
		val, err := dc.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			var configs map[string]interface{}
			if err := json.Unmarshal([]byte(val), &configs); err == nil {
				if dc.logger != nil {
					dc.logger.Printf("Domain cache hit (Redis)")
				}
				return configs, nil
			}
		}
	}

	// Fallback to in-memory cache
	dc.mu.RLock()
	cached, exists := dc.inMemory[cacheKey]
	dc.mu.RUnlock()

	if exists && time.Now().Before(cached.ExpiresAt) {
		if dc.logger != nil {
			dc.logger.Printf("Domain cache hit (in-memory)")
		}
		return cached.Config, nil
	}

	return nil, fmt.Errorf("cache miss")
}

// Set stores domain configurations in cache
func (dc *DomainCache) Set(ctx context.Context, configs map[string]interface{}) error {
	cacheKey := "domain:configs:all"
	expiresAt := time.Now().Add(dc.ttl)

	// Store in Redis if available
	if dc.redisClient != nil {
		data, err := json.Marshal(configs)
		if err == nil {
			err = dc.redisClient.Set(ctx, cacheKey, data, dc.ttl).Err()
			if err == nil {
				if dc.logger != nil {
					dc.logger.Printf("Domain configs cached in Redis (TTL: %v)", dc.ttl)
				}
			}
		}
	}

	// Also store in-memory as fallback
	dc.mu.Lock()
	dc.inMemory[cacheKey] = cachedDomain{
		Config:    configs,
		ExpiresAt: expiresAt,
	}
	dc.mu.Unlock()

	return nil
}

// Invalidate clears the cache
func (dc *DomainCache) Invalidate(ctx context.Context) error {
	cacheKey := "domain:configs:all"

	if dc.redisClient != nil {
		dc.redisClient.Del(ctx, cacheKey)
	}

	dc.mu.Lock()
	delete(dc.inMemory, cacheKey)
	dc.mu.Unlock()

	if dc.logger != nil {
		dc.logger.Printf("Domain cache invalidated")
	}

	return nil
}

// CleanupExpired removes expired entries from in-memory cache
func (dc *DomainCache) CleanupExpired() {
	dc.mu.Lock()
	defer dc.mu.Unlock()

	now := time.Now()
	for key, cached := range dc.inMemory {
		if now.After(cached.ExpiresAt) {
			delete(dc.inMemory, key)
		}
	}
}

