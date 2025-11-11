package redisstore

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/langchain-ai/langgraph-go/pkg/checkpoint"
)

const defaultPrefix = "langgraph:checkpoint:"

// Option mutates store configuration.
type Option func(*Store)

// WithPrefix overrides the Redis key prefix used for checkpoints.
func WithPrefix(prefix string) Option {
	return func(s *Store) {
		if prefix != "" {
			s.prefix = prefix
		}
	}
}

// WithTTL sets an expiration for stored checkpoints. Zero duration keeps data indefinitely.
func WithTTL(ttl time.Duration) Option {
	return func(s *Store) {
		s.ttl = ttl
	}
}

// Store implements checkpoint persistence on top of Redis.
type Store struct {
	client *redis.Client
	prefix string
	ttl    time.Duration
}

// NewStore builds a Store from an existing Redis client. The client must not be nil.
func NewStore(client *redis.Client, opts ...Option) (*Store, error) {
	if client == nil {
		return nil, fmt.Errorf("redis checkpoint: client is nil")
	}
	store := &Store{
		client: client,
		prefix: defaultPrefix,
	}
	for _, opt := range opts {
		opt(store)
	}
	// Validate connectivity eagerly so misconfiguration is caught early.
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("redis checkpoint: ping failed: %w", err)
	}
	return store, nil
}

func (s *Store) key(id string) string {
	return s.prefix + id
}

// Save writes the payload under the supplied key.
func (s *Store) Save(ctx context.Context, key string, payload []byte) error {
	if err := s.client.Set(ctx, s.key(key), payload, s.ttl).Err(); err != nil {
		return fmt.Errorf("redis checkpoint: set %q: %w", key, err)
	}
	return nil
}

// Load retrieves the stored payload. checkpoint.ErrNotFound is returned when the key is absent.
func (s *Store) Load(ctx context.Context, key string) ([]byte, error) {
	val, err := s.client.Get(ctx, s.key(key)).Bytes()
	if err == redis.Nil {
		return nil, checkpoint.ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("redis checkpoint: get %q: %w", key, err)
	}
	return val, nil
}

// Delete removes the stored payload. Missing keys are treated as success.
func (s *Store) Delete(ctx context.Context, key string) error {
	if err := s.client.Del(ctx, s.key(key)).Err(); err != nil {
		return fmt.Errorf("redis checkpoint: delete %q: %w", key, err)
	}
	return nil
}
