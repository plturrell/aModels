package domain

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisConfigLoader loads domain configurations from Redis
type RedisConfigLoader struct {
	client *redis.Client
	key    string
}

// NewRedisConfigLoader creates a new Redis-based config loader
func NewRedisConfigLoader(redisURL, key string) (*RedisConfigLoader, error) {
	// Parse Redis URL (format: redis://[:password@]host[:port][/db])
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		// Fallback: try simple host:port format
		if redisURL == "" {
			redisURL = "redis://localhost:6379/0"
		} else if redisURL[0] != 'r' || !contains(redisURL, "://") {
			// Assume it's just host:port
			redisURL = "redis://" + redisURL + "/0"
		}
		opts, err = redis.ParseURL(redisURL)
		if err != nil {
			return nil, fmt.Errorf("parse redis URL: %w", err)
		}
	}

	client := redis.NewClient(opts)
	
	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis connection failed: %w", err)
	}

	if key == "" {
		key = "localai:domains:config"
	}

	return &RedisConfigLoader{
		client: client,
		key:    key,
	}, nil
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || containsMid(s, substr)))
}

func containsMid(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// LoadDomainConfigs loads domain configurations from Redis
func (r *RedisConfigLoader) LoadDomainConfigs(ctx context.Context, dm *DomainManager) error {
	data, err := r.client.Get(ctx, r.key).Result()
	if err == redis.Nil {
		return fmt.Errorf("domain config not found in Redis (key: %s)", r.key)
	}
	if err != nil {
		return fmt.Errorf("redis get error: %w", err)
	}

	var config DomainsConfig
	if err := json.Unmarshal([]byte(data), &config); err != nil {
		return fmt.Errorf("parse config JSON: %w", err)
	}

	filtered := make(map[string]*DomainConfig)
	for name, cfg := range config.Domains {
		if cfg == nil {
			continue
		}
		if !isDomainEnabled(cfg.EnabledEnvVar) {
			continue
		}
		if err := cfg.Validate(); err != nil {
			log.Printf("⚠️  Domain %s invalid: %v", name, err)
			continue
		}
		filtered[name] = cfg
	}

	if len(filtered) == 0 {
		return fmt.Errorf("no domains enabled after applying configuration toggles")
	}

	dm.mu.Lock()
	defer dm.mu.Unlock()

	dm.domains = filtered

	newDefault := dm.defaultDomain
	if config.DefaultDomain != "" {
		newDefault = config.DefaultDomain
	}
	if _, exists := filtered[newDefault]; !exists {
		newDefault = ""
	}
	if newDefault == "" {
		if _, exists := filtered["general"]; exists {
			newDefault = "general"
		} else {
			for name := range filtered {
				newDefault = name
				break
			}
		}
	}
	if newDefault == "" {
		return fmt.Errorf("could not determine default domain after applying configuration toggles")
	}
	dm.defaultDomain = newDefault

	log.Printf("✅ Loaded %d domains from Redis", len(filtered))
	return nil
}

// WatchConfig watches for configuration changes in Redis
func (r *RedisConfigLoader) WatchConfig(ctx context.Context, dm *DomainManager, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := r.LoadDomainConfigs(ctx, dm); err != nil {
				log.Printf("⚠️  Failed to reload config from Redis: %v", err)
			} else {
				log.Printf("🔄 Reloaded domain configs from Redis")
			}
		}
	}
}

// Close closes the Redis connection
func (r *RedisConfigLoader) Close() error {
	return r.client.Close()
}

