package storage

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

// SaveVector saves a vector to Redis with metadata.
func (p *RedisPersistence) SaveVector(key string, vector []float32, metadata map[string]any) error {
	ctx := context.Background()

	// Serialize the vector to JSON
	jsonVector, err := json.Marshal(vector)
	if err != nil {
		return fmt.Errorf("failed to marshal vector: %w", err)
	}

	// Serialize metadata to JSON
	jsonMetadata := "{}"
	if metadata != nil {
		metadataBytes, err := json.Marshal(metadata)
		if err != nil {
			return fmt.Errorf("failed to marshal metadata: %w", err)
		}
		jsonMetadata = string(metadataBytes)
	}

	// Save the vector and metadata to Redis hash
	err = p.client.HSet(ctx, key, map[string]any{
		"vector":   jsonVector,
		"metadata": jsonMetadata,
	}).Err()
	if err != nil {
		return fmt.Errorf("failed to save vector: %w", err)
	}

	return nil
}

// GetVector retrieves a vector and metadata from Redis.
func (p *RedisPersistence) GetVector(key string) ([]float32, map[string]any, error) {
	ctx := context.Background()

	result, err := p.client.HMGet(ctx, key, "vector", "metadata").Result()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get vector: %w", err)
	}

	if len(result) < 2 || result[0] == nil {
		return nil, nil, fmt.Errorf("vector not found: %s", key)
	}

	// Parse vector
	vectorStr, ok := result[0].(string)
	if !ok {
		return nil, nil, fmt.Errorf("invalid vector format")
	}
	var vector []float32
	if err := json.Unmarshal([]byte(vectorStr), &vector); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal vector: %w", err)
	}

	// Parse metadata
	metadata := make(map[string]any)
	if result[1] != nil {
		metadataStr, ok := result[1].(string)
		if ok && metadataStr != "" {
			if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
				// Non-fatal: return empty metadata if unmarshal fails
				metadata = make(map[string]any)
			}
		}
	}

	return vector, metadata, nil
}

// SearchSimilar performs basic similarity search using Redis SCAN and cosine similarity.
// This is a fallback implementation - for production use pgvector or OpenSearch.
func (p *RedisPersistence) SearchSimilar(queryVector []float32, artifactType string, limit int, threshold float32) ([]VectorSearchResult, error) {
	ctx := context.Background()

	// SCAN for all keys matching pattern
	pattern := "*"
	if artifactType != "" {
		pattern = fmt.Sprintf("%s:*", artifactType)
	}

	var allKeys []string
	iter := p.client.Scan(ctx, 0, pattern, 0).Iterator()
	for iter.Next(ctx) {
		allKeys = append(allKeys, iter.Val())
	}
	if err := iter.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan keys: %w", err)
	}

	// Calculate cosine similarity for each vector
	type scoredResult struct {
		result VectorSearchResult
		score  float32
	}
	scoredResults := []scoredResult{}

	for _, key := range allKeys {
		vector, metadata, err := p.GetVector(key)
		if err != nil {
			continue // Skip keys without vectors
		}

		// Filter by artifact type if specified
		if artifactType != "" {
			if metaType, ok := metadata["artifact_type"].(string); !ok || metaType != artifactType {
				continue
			}
		}

		// Calculate cosine similarity
		score := float32(cosineSimilarity(queryVector, vector))
		if score >= threshold {
			artifactType := ""
			artifactID := ""
			text := ""
			if metadata != nil {
				if at, ok := metadata["artifact_type"].(string); ok {
					artifactType = at
				}
				if aid, ok := metadata["artifact_id"].(string); ok {
					artifactID = aid
				}
				if t, ok := metadata["label"].(string); ok {
					text = t
				}
			}

			scoredResults = append(scoredResults, scoredResult{
				result: VectorSearchResult{
					Key:          key,
					ArtifactType: artifactType,
					ArtifactID:   artifactID,
					Vector:       vector,
					Metadata:     metadata,
					Score:        score,
					Text:         text,
				},
				score: score,
			})
		}
	}

	// Sort by score (descending) and limit
	// Simple bubble sort for small datasets (optimize later if needed)
	for i := 0; i < len(scoredResults)-1; i++ {
		for j := i + 1; j < len(scoredResults); j++ {
			if scoredResults[i].score < scoredResults[j].score {
				scoredResults[i], scoredResults[j] = scoredResults[j], scoredResults[i]
			}
		}
	}

	if limit > 0 && limit < len(scoredResults) {
		scoredResults = scoredResults[:limit]
	}

	results := make([]VectorSearchResult, len(scoredResults))
	for i, sr := range scoredResults {
		results[i] = sr.result
	}

	return results, nil
}

// SearchByText performs text-based search using Redis SCAN and metadata matching.
// This is a basic implementation - for production use OpenSearch.
func (p *RedisPersistence) SearchByText(query string, artifactType string, limit int) ([]VectorSearchResult, error) {
	ctx := context.Background()

	pattern := "*"
	if artifactType != "" {
		pattern = fmt.Sprintf("%s:*", artifactType)
	}

	var allKeys []string
	iter := p.client.Scan(ctx, 0, pattern, 0).Iterator()
	for iter.Next(ctx) {
		allKeys = append(allKeys, iter.Val())
	}
	if err := iter.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan keys: %w", err)
	}

	queryLower := strings.ToLower(query)
	results := []VectorSearchResult{}

	for _, key := range allKeys {
		vector, metadata, err := p.GetVector(key)
		if err != nil {
			continue
		}

		// Filter by artifact type if specified
		if artifactType != "" {
			if metaType, ok := metadata["artifact_type"].(string); !ok || metaType != artifactType {
				continue
			}
		}

		// Simple text matching in label/metadata
		matched := false
		if label, ok := metadata["label"].(string); ok {
			if strings.Contains(strings.ToLower(label), queryLower) {
				matched = true
			}
		}

		if !matched && metadata != nil {
			// Check other text fields
			for _, value := range metadata {
				if str, ok := value.(string); ok {
					if strings.Contains(strings.ToLower(str), queryLower) {
						matched = true
						break
					}
				}
			}
		}

		if matched {
			artifactType := ""
			artifactID := ""
			text := ""
			if metadata != nil {
				if at, ok := metadata["artifact_type"].(string); ok {
					artifactType = at
				}
				if aid, ok := metadata["artifact_id"].(string); ok {
					artifactID = aid
				}
				if t, ok := metadata["label"].(string); ok {
					text = t
				}
			}

			results = append(results, VectorSearchResult{
				Key:          key,
				ArtifactType: artifactType,
				ArtifactID:   artifactID,
				Vector:       vector,
				Metadata:     metadata,
				Score:        1.0, // Text match doesn't have similarity score
				Text:         text,
			})

			if limit > 0 && len(results) >= limit {
				break
			}
		}
	}

	return results, nil
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
