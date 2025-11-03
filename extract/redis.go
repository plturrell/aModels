package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisPersistence is the persistence layer for Redis.
type RedisPersistence struct {
	client *redis.Client
}

// NewRedisPersistence creates a new Redis persistence layer.
func NewRedisPersistence(addr, password string, db int) (*RedisPersistence, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	// Ping the server to check the connection.
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to redis: %w", err)
	}

	return &RedisPersistence{client: client}, nil
}

// SaveVector saves a vector to Redis.
func (p *RedisPersistence) SaveVector(key string, vector []float32) error {
	// Serialize the vector to JSON
	jsonVector, err := json.Marshal(vector)
	if err != nil {
		return fmt.Errorf("failed to marshal vector: %w", err)
	}

	// Save the vector to Redis
	err = p.client.HSet(context.Background(), key, "vector", jsonVector).Err()
	if err != nil {
		return fmt.Errorf("failed to save vector: %w", err)
	}

	return nil
}

// SaveSchema stores graph nodes and edges as JSON payloads under deterministic keys.
func (p *RedisPersistence) SaveSchema(nodes []Node, edges []Edge) error {
	ctx := context.Background()
	pipe := p.client.Pipeline()

	for _, node := range nodes {
		if strings.TrimSpace(node.ID) == "" {
			continue
		}
		payload := map[string]any{
			"id":          node.ID,
			"type":        node.Type,
			"label":       node.Label,
			"properties":  node.Props,
			"recorded_at": time.Now().UTC().Format(time.RFC3339Nano),
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("failed to marshal redis node %s: %w", node.ID, err)
		}
		pipe.Set(ctx, fmt.Sprintf("glean:nodes:%s", node.ID), data, 0)
	}

	for _, edge := range edges {
		if strings.TrimSpace(edge.SourceID) == "" || strings.TrimSpace(edge.TargetID) == "" {
			continue
		}
		payload := map[string]any{
			"source":      edge.SourceID,
			"target":      edge.TargetID,
			"label":       edge.Label,
			"properties":  edge.Props,
			"recorded_at": time.Now().UTC().Format(time.RFC3339Nano),
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("failed to marshal redis edge %s->%s: %w", edge.SourceID, edge.TargetID, err)
		}
		key := fmt.Sprintf("glean:edges:%s:%s:%s", edge.SourceID, edge.TargetID, edge.Label)
		pipe.Set(ctx, key, data, 0)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to persist glean schema to redis: %w", err)
	}
	return nil
}
