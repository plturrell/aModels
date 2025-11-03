package cli

import (
	"database/sql"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/redis/go-redis/v9"

	hanastore "github.com/langchain-ai/langgraph-go/pkg/checkpoint/hana"
	redisstore "github.com/langchain-ai/langgraph-go/pkg/checkpoint/redis"
	sqlitestore "github.com/langchain-ai/langgraph-go/pkg/checkpoint/sqlite"
	"github.com/langchain-ai/langgraph-go/pkg/graph"
	hanapool "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
)

// BuildStateManager constructs a state manager for the requested checkpoint
// backend. The returned cleanup function should be deferred by the caller.
func BuildStateManager(spec string) (*graph.StateManager, func() error, error) {
	spec = strings.TrimSpace(spec)
	switch {
	case spec == "":
		return nil, nil, fmt.Errorf("checkpoint backend must be specified (hana, hana+blockchain, redis, sqlite:/path)")
	case strings.HasPrefix(spec, "sqlite:"):
		path := strings.TrimPrefix(spec, "sqlite:")
		if path == "" {
			return nil, nil, fmt.Errorf("sqlite checkpoint requires a file path")
		}
		db, err := sql.Open("sqlite3", path)
		if err != nil {
			return nil, nil, fmt.Errorf("open sqlite checkpoint db: %w", err)
		}
		store, err := sqlitestore.NewStore(db)
		if err != nil {
			_ = db.Close()
			return nil, nil, err
		}
		return graph.NewStateManager(store), db.Close, nil
	case spec == "hana":
		pool, err := hanapool.NewPoolFromEnv()
		if err != nil {
			return nil, nil, fmt.Errorf("create hana pool: %w", err)
		}
		store, err := hanastore.NewStore(pool)
		if err != nil {
			_ = pool.Close()
			return nil, nil, err
		}
		return graph.NewStateManager(store), pool.Close, nil
	case strings.HasPrefix(spec, "hana+"):
		mode := strings.TrimPrefix(spec, "hana+")
		return buildBlockchainCheckpointManager(mode)
	case strings.HasPrefix(spec, "redis://"):
		opts, err := redis.ParseURL(spec)
		if err != nil {
			return nil, nil, fmt.Errorf("parse redis url: %w", err)
		}
		client := redis.NewClient(opts)
		store, err := redisstore.NewStore(client, redisstore.WithPrefix(defaultRedisPrefix()))
		if err != nil {
			_ = client.Close()
			return nil, nil, err
		}
		return graph.NewStateManager(store), client.Close, nil
	case spec == "redis":
		client, prefix, ttl, err := newRedisClientFromEnv()
		if err != nil {
			return nil, nil, err
		}
		opts := []redisstore.Option{redisstore.WithPrefix(prefix)}
		if ttl > 0 {
			opts = append(opts, redisstore.WithTTL(ttl))
		}
		store, err := redisstore.NewStore(client, opts...)
		if err != nil {
			_ = client.Close()
			return nil, nil, err
		}
		return graph.NewStateManager(store), client.Close, nil
	default:
		return nil, nil, fmt.Errorf("unknown checkpoint backend %q", spec)
	}
}

func defaultRedisPrefix() string {
	if prefix := os.Getenv("REDIS_PREFIX"); prefix != "" {
		return prefix
	}
	return "langgraph:checkpoint:"
}

func newRedisClientFromEnv() (*redis.Client, string, time.Duration, error) {
	addr := os.Getenv("REDIS_ADDR")
	if strings.TrimSpace(addr) == "" {
		addr = "localhost:6379"
	}
	db := 0
	if raw := os.Getenv("REDIS_DB"); raw != "" {
		if v, err := strconv.Atoi(raw); err == nil {
			db = v
		} else {
			return nil, "", 0, fmt.Errorf("parse REDIS_DB: %w", err)
		}
	}
	ttl := time.Duration(0)
	if raw := os.Getenv("REDIS_TTL_SECONDS"); raw != "" {
		seconds, err := strconv.Atoi(raw)
		if err != nil {
			return nil, "", 0, fmt.Errorf("parse REDIS_TTL_SECONDS: %w", err)
		}
		ttl = time.Duration(seconds) * time.Second
	}
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: os.Getenv("REDIS_PASSWORD"),
		DB:       db,
	})
	return client, defaultRedisPrefix(), ttl, nil
}

// buildBlockchainCheckpointManager is implemented in state_blockchain.go when the
// binary is built with the hana,blockchain build tags. The default implementation
// returns an informative error.
func buildBlockchainCheckpointManager(mode string) (*graph.StateManager, func() error, error) {
	return nil, nil, fmt.Errorf("checkpoint backend hana+%s requires build tags 'hana,blockchain'", mode)
}
