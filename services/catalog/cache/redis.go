package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// Logger interface for cache logging
type Logger interface {
	Info(msg string, fields map[string]interface{})
	Debug(msg string, fields map[string]interface{})
}

// Cache provides Redis-based caching.
type Cache struct {
	client *redis.Client
	logger Logger
}

// NewCache creates a new Redis cache.
func NewCache(redisURL string, logger Logger) (*Cache, error) {
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	client := redis.NewClient(opt)

	// Test connection
	ctx := context.Background()
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &Cache{
		client: client,
		logger: logger,
	}, nil
}

// Get retrieves a value from cache.
func (c *Cache) Get(ctx context.Context, key string, dest interface{}) error {
	val, err := c.client.Get(ctx, key).Result()
	if err == redis.Nil {
		if c.logger != nil {
			c.logger.Debug("Cache miss", map[string]interface{}{"key": key})
		}
		return ErrCacheMiss
	}
	if err != nil {
		return fmt.Errorf("failed to get from cache: %w", err)
	}

	if c.logger != nil {
		c.logger.Debug("Cache hit", map[string]interface{}{"key": key})
	}

	if err := json.Unmarshal([]byte(val), dest); err != nil {
		return fmt.Errorf("failed to unmarshal cached value: %w", err)
	}

	return nil
}

// Set stores a value in cache.
func (c *Cache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %w", err)
	}

	if err := c.client.Set(ctx, key, data, ttl).Err(); err != nil {
		return fmt.Errorf("failed to set cache: %w", err)
	}

	if c.logger != nil {
		c.logger.Debug("Cache set", map[string]interface{}{
			"key": key,
			"ttl": ttl.Seconds(),
		})
	}

	return nil
}

// Delete removes a value from cache.
func (c *Cache) Delete(ctx context.Context, key string) error {
	if err := c.client.Del(ctx, key).Err(); err != nil {
		return fmt.Errorf("failed to delete from cache: %w", err)
	}

	if c.logger != nil {
		c.logger.Debug("Cache delete", map[string]interface{}{"key": key})
	}

	return nil
}

// Exists checks if a key exists in cache.
func (c *Cache) Exists(ctx context.Context, key string) (bool, error) {
	count, err := c.client.Exists(ctx, key).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check cache existence: %w", err)
	}
	return count > 0, nil
}

// Close closes the cache connection.
func (c *Cache) Close() error {
	return c.client.Close()
}

// ErrCacheMiss is returned when a cache key is not found.
var ErrCacheMiss = fmt.Errorf("cache miss")

// CacheKey generates a cache key from components.
func CacheKey(components ...string) string {
	return fmt.Sprintf("catalog:%s", fmt.Sprintf("%s", components))
}

