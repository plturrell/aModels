package main

import (
	"fmt"
	"log"
)

// CompositeVectorPersistence combines multiple vector persistence implementations.
// Writes go to all stores, reads/search use the primary store (pgvector for structured, OpenSearch for semantic).
type CompositeVectorPersistence struct {
	primary   VectorPersistence // Primary store (pgvector for reads)
	secondary []VectorPersistence // Secondary stores (OpenSearch for search, Redis for cache)
	logger    *log.Logger
}

// NewCompositeVectorPersistence creates a composite vector persistence.
func NewCompositeVectorPersistence(primary VectorPersistence, secondary []VectorPersistence, logger *log.Logger) *CompositeVectorPersistence {
	return &CompositeVectorPersistence{
		primary:   primary,
		secondary: secondary,
		logger:    logger,
	}
}

// SaveVector saves to all stores (primary and secondary).
func (c *CompositeVectorPersistence) SaveVector(key string, vector []float32, metadata map[string]any) error {
	// Save to primary
	if c.primary != nil {
		if err := c.primary.SaveVector(key, vector, metadata); err != nil {
			c.logger.Printf("failed to save vector to primary store: %v", err)
			// Continue to secondary stores even if primary fails
		}
	}

	// Save to secondary stores
	var errs []error
	for _, store := range c.secondary {
		if store != nil {
			if err := store.SaveVector(key, vector, metadata); err != nil {
				c.logger.Printf("failed to save vector to secondary store: %v", err)
				errs = append(errs, err)
			}
		}
	}

	// Return error only if all stores failed
	if c.primary == nil && len(errs) == len(c.secondary) {
		if len(errs) > 0 {
			return errs[0]
		}
		return nil
	}

	return nil
}

// GetVector retrieves from primary store (pgvector).
func (c *CompositeVectorPersistence) GetVector(key string) ([]float32, map[string]any, error) {
	if c.primary != nil {
		return c.primary.GetVector(key)
	}

	// Fallback to first secondary store
	if len(c.secondary) > 0 && c.secondary[0] != nil {
		return c.secondary[0].GetVector(key)
	}

	return nil, nil, fmt.Errorf("no vector store available")
}

// SearchSimilar searches using OpenSearch if available, otherwise falls back to primary.
func (c *CompositeVectorPersistence) SearchSimilar(queryVector []float32, artifactType string, limit int, threshold float32) ([]VectorSearchResult, error) {
	// Try OpenSearch first (best for semantic search)
	for _, store := range c.secondary {
		if _, ok := store.(*OpenSearchPersistence); ok {
			// OpenSearch has its own SearchSimilar implementation
			return store.SearchSimilar(queryVector, artifactType, limit, threshold)
		}
	}

	// Fallback to primary (pgvector)
	if c.primary != nil {
		return c.primary.SearchSimilar(queryVector, artifactType, limit, threshold)
	}

	// Fallback to any secondary store
	for _, store := range c.secondary {
		if store != nil {
			return store.SearchSimilar(queryVector, artifactType, limit, threshold)
		}
	}

	return nil, fmt.Errorf("no vector store available for search")
}

// SearchByText searches using OpenSearch if available, otherwise falls back to primary.
func (c *CompositeVectorPersistence) SearchByText(query string, artifactType string, limit int) ([]VectorSearchResult, error) {
	// Try OpenSearch first (best for text search)
	for _, store := range c.secondary {
		if opensearch, ok := store.(*OpenSearchPersistence); ok {
			return opensearch.SearchByText(query, artifactType, limit)
		}
	}

	// Fallback to primary (pgvector)
	if c.primary != nil {
		return c.primary.SearchByText(query, artifactType, limit)
	}

	// Fallback to any secondary store
	for _, store := range c.secondary {
		if store != nil {
			return store.SearchByText(query, artifactType, limit)
		}
	}

	return nil, fmt.Errorf("no vector store available for text search")
}

