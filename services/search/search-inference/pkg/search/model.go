package search

import (
	"context"
)

// SearchModel provides a stub implementation for embedding and rerank tasks.
// This is a fallback implementation used when LocalAI is not available.
// In production, SearchModelWithLocalAI (which wraps this) should be used instead,
// as it provides actual embedding capabilities via LocalAI.
//
// The stub methods return empty/zero results and are intended only for:
// - Development/testing environments without LocalAI
// - Graceful degradation when LocalAI is unavailable
// - Fallback behavior in SearchModelWithLocalAI when embedder is nil

type SearchModel struct {
	baseModel interface{} // Stub - LocalAI not available in standalone repo
}

// LoadSearchModel creates a stub SearchModel instance.
// This is a no-op implementation that always succeeds.
// For production use, prefer NewSearchModelWithLocalAI which provides
// actual embedding capabilities via LocalAI.
func LoadSearchModel(modelPath string) (*SearchModel, error) {
	// LocalAI dependency removed - return stub model
	// This stub is only used as a fallback when LocalAI embedder is unavailable
	return &SearchModel{baseModel: nil}, nil
}

// Close releases any resources held by the model.
// This is a no-op for the stub implementation.
func (s *SearchModel) Close() error {
	return nil
}

// Embed generates an embedding for the given text.
// STUB IMPLEMENTATION: Returns an empty embedding vector.
// This method should not be called directly in production code.
// Use SearchModelWithLocalAI.Embed() instead, which delegates to LocalAI.
//
// When used as a fallback, this returns an empty slice, which will result
// in zero similarity scores in search results.
func (s *SearchModel) Embed(ctx context.Context, text string) ([]float64, error) {
	// Stub implementation: return empty embedding
	// In production, this should never be called when LocalAI is available
	// The SearchModelWithLocalAI wrapper uses LocalAIEmbedder instead
	_ = ctx
	_ = text
	return []float64{}, nil
}

// Rerank reranks documents based on their relevance to the query.
// STUB IMPLEMENTATION: Returns zero scores for all documents.
// This method should not be called directly in production code.
// Use SearchModelWithLocalAI.Rerank() instead.
//
// When used as a fallback, this returns zero scores for all documents,
// effectively disabling reranking functionality.
func (s *SearchModel) Rerank(ctx context.Context, query string, documents []string) ([]float64, error) {
	// Stub implementation: return zero scores for all documents
	// In production, this should never be called when LocalAI is available
	_ = ctx
	_ = query
	if len(documents) == 0 {
		return []float64{}, nil
	}
	// Return zero scores - documents will not be reranked
	return make([]float64, len(documents)), nil
}
