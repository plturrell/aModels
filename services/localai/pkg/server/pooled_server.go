package server

import (
	"context"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/cache"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/logging"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/pool"
)

// PooledServer wraps VaultGemmaServer with pooling and caching enhancements
type PooledServer struct {
	*VaultGemmaServer
	logger      *logging.Logger
	dbPool      *pool.DBPool
	httpPool    *pool.HTTPPool
	inferenceCache *cache.LRUCache
	embeddingCache *cache.LRUCache
}

// PooledServerConfig holds configuration for the pooled server
type PooledServerConfig struct {
	// Database pooling
	PostgresDSN      string
	DBMaxOpenConns   int
	DBMaxIdleConns   int
	
	// HTTP client pooling
	HTTPMaxIdleConns        int
	HTTPMaxIdleConnsPerHost int
	HTTPTimeout             time.Duration
	
	// Cache configuration
	InferenceCacheCapacity int
	InferenceCacheTTL      time.Duration
	EmbeddingCacheCapacity int
	EmbeddingCacheTTL      time.Duration
	
	// Logging
	LogLevel  string
	LogFormat string
}

// DefaultPooledServerConfig returns sensible defaults
func DefaultPooledServerConfig() *PooledServerConfig {
	return &PooledServerConfig{
		DBMaxOpenConns:          25,
		DBMaxIdleConns:          5,
		HTTPMaxIdleConns:        100,
		HTTPMaxIdleConnsPerHost: 10,
		HTTPTimeout:             30 * time.Second,
		InferenceCacheCapacity:  1000,
		InferenceCacheTTL:       1 * time.Hour,
		EmbeddingCacheCapacity:  500,
		EmbeddingCacheTTL:       24 * time.Hour,
		LogLevel:                "info",
		LogFormat:               "json",
	}
}

// NewPooledServer creates a new server with pooling and caching
func NewPooledServer(vgServer *VaultGemmaServer, cfg *PooledServerConfig) (*PooledServer, error) {
	if cfg == nil {
		cfg = DefaultPooledServerConfig()
	}

	// Initialize structured logger
	logger := logging.NewLogger(&logging.Config{
		Level:  cfg.LogLevel,
		Format: cfg.LogFormat,
	})

	pooledServer := &PooledServer{
		VaultGemmaServer: vgServer,
		logger:           logger,
	}

	// Initialize database pool if DSN provided
	if cfg.PostgresDSN != "" {
		dbPool, err := pool.NewDBPool(&pool.DBPoolConfig{
			DSN:             cfg.PostgresDSN,
			MaxOpenConns:    cfg.DBMaxOpenConns,
			MaxIdleConns:    cfg.DBMaxIdleConns,
			ConnMaxLifetime: 30 * time.Minute,
			ConnMaxIdleTime: 5 * time.Minute,
		})
		if err != nil {
			logger.Error("Failed to initialize database pool", err, nil)
		} else {
			pooledServer.dbPool = dbPool
			logger.Info("Database connection pool initialized", map[string]interface{}{
				"max_open_conns": cfg.DBMaxOpenConns,
				"max_idle_conns": cfg.DBMaxIdleConns,
			})
		}
	}

	// Initialize HTTP client pool
	pooledServer.httpPool = pool.NewHTTPPool(&pool.HTTPPoolConfig{
		MaxIdleConns:        cfg.HTTPMaxIdleConns,
		MaxIdleConnsPerHost: cfg.HTTPMaxIdleConnsPerHost,
		Timeout:             cfg.HTTPTimeout,
	})
	logger.Info("HTTP connection pool initialized", map[string]interface{}{
		"max_idle_conns": cfg.HTTPMaxIdleConns,
		"timeout_seconds": cfg.HTTPTimeout.Seconds(),
	})

	// Initialize inference cache
	pooledServer.inferenceCache = cache.NewLRUCache(&cache.LRUConfig{
		Capacity:   cfg.InferenceCacheCapacity,
		DefaultTTL: cfg.InferenceCacheTTL,
		OnEvict: func(key string, value interface{}) {
			logger.Debug("Cache entry evicted", map[string]interface{}{
				"key":  key,
				"type": "inference",
			})
		},
	})
	logger.Info("Inference cache initialized", map[string]interface{}{
		"capacity": cfg.InferenceCacheCapacity,
		"ttl":      cfg.InferenceCacheTTL.String(),
	})

	// Initialize embedding cache
	pooledServer.embeddingCache = cache.NewLRUCache(&cache.LRUConfig{
		Capacity:   cfg.EmbeddingCacheCapacity,
		DefaultTTL: cfg.EmbeddingCacheTTL,
		OnEvict: func(key string, value interface{}) {
			logger.Debug("Cache entry evicted", map[string]interface{}{
				"key":  key,
				"type": "embedding",
			})
		},
	})
	logger.Info("Embedding cache initialized", map[string]interface{}{
		"capacity": cfg.EmbeddingCacheCapacity,
		"ttl":      cfg.EmbeddingCacheTTL.String(),
	})

	// Start cache cleanup timers
	ctx := context.Background()
	pooledServer.inferenceCache.StartCleanupTimer(ctx, 5*time.Minute)
	pooledServer.embeddingCache.StartCleanupTimer(ctx, 10*time.Minute)

	return pooledServer, nil
}

// GetLogger returns the structured logger
func (s *PooledServer) GetLogger() *logging.Logger {
	return s.logger
}

// GetDBPool returns the database connection pool
func (s *PooledServer) GetDBPool() *pool.DBPool {
	return s.dbPool
}

// GetHTTPPool returns the HTTP client pool
func (s *PooledServer) GetHTTPPool() *pool.HTTPPool {
	return s.httpPool
}

// GetInferenceCache returns the inference cache
func (s *PooledServer) GetInferenceCache() *cache.LRUCache {
	return s.inferenceCache
}

// GetEmbeddingCache returns the embedding cache
func (s *PooledServer) GetEmbeddingCache() *cache.LRUCache {
	return s.embeddingCache
}

// GetPoolStats returns statistics for all pools
func (s *PooledServer) GetPoolStats() map[string]interface{} {
	stats := make(map[string]interface{})

	if s.dbPool != nil {
		stats["db_pool"] = s.dbPool.GetStats()
	}

	if s.httpPool != nil {
		stats["http_pool"] = s.httpPool.GetStats()
	}

	if s.inferenceCache != nil {
		stats["inference_cache"] = s.inferenceCache.GetStats()
	}

	if s.embeddingCache != nil {
		stats["embedding_cache"] = s.embeddingCache.GetStats()
	}

	return stats
}

// Close gracefully closes all pools and connections
func (s *PooledServer) Close() error {
	s.logger.Info("Shutting down pooled server", nil)

	if s.dbPool != nil {
		if err := s.dbPool.Close(); err != nil {
			s.logger.Error("Failed to close database pool", err, nil)
		}
	}

	if s.httpPool != nil {
		s.httpPool.Close()
	}

	if s.inferenceCache != nil {
		ctx := context.Background()
		s.inferenceCache.Clear(ctx)
	}

	if s.embeddingCache != nil {
		ctx := context.Background()
		s.embeddingCache.Clear(ctx)
	}

	s.logger.Info("Pooled server shutdown complete", nil)
	return nil
}
