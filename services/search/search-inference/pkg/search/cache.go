package search

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisConfig defines connection parameters for the embedding cache.
type RedisConfig struct {
	Addr      string
	Password  string
	DB        int
	TLSConfig *tls.Config
}

type vectorCache struct {
	client redis.UniversalClient
	ttl    time.Duration
}

func newVectorCache(cfg RedisConfig, ttl time.Duration) (*vectorCache, error) {
	// Redis is optional - if not provided, use in-memory fallback
	if cfg.Addr == "" {
		// Return nil cache which will use in-memory map fallback
		return nil, nil
	}

	client := redis.NewClient(&redis.Options{
		Addr:      cfg.Addr,
		Password:  cfg.Password,
		DB:        cfg.DB,
		TLSConfig: cfg.TLSConfig,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		// If Redis connection fails, log warning but continue without cache
		client.Close()
		fmt.Printf("⚠️  Redis unavailable at %s, using in-memory cache: %v\n", cfg.Addr, err)
		return nil, nil
	}

	if ttl <= 0 {
		ttl = 30 * time.Minute
	}

	return &vectorCache{client: client, ttl: ttl}, nil
}

func (vc *vectorCache) Close() error {
	if vc == nil || vc.client == nil {
		return nil
	}
	return vc.client.Close()
}

func (vc *vectorCache) cacheKey(text string) string {
	sum := sha256.Sum256([]byte(text))
	return "embedding:" + hex.EncodeToString(sum[:])
}

func (vc *vectorCache) Get(ctx context.Context, text string) ([]float64, bool) {
	if vc == nil || vc.client == nil {
		return nil, false
	}

	raw, err := vc.client.Get(ctx, vc.cacheKey(text)).Bytes()
	if err != nil {
		return nil, false
	}

	var embedding []float64
	if err := json.Unmarshal(raw, &embedding); err != nil {
		return nil, false
	}

	return embedding, true
}

func (vc *vectorCache) Set(ctx context.Context, text string, embedding []float64) {
	if vc == nil || vc.client == nil {
		return
	}

	payload, err := json.Marshal(embedding)
	if err != nil {
		return
	}

	_ = vc.client.Set(ctx, vc.cacheKey(text), payload, vc.ttl).Err()
}
