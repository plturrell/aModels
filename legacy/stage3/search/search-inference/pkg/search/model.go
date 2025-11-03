package search

import (
	"context"
)

// SearchModel wraps VaultGemma for embedding and rerank tasks
// Reuses the baseline weights from training workspace
// Note: LocalAI models package not available - using stub implementation

type SearchModel struct {
	baseModel interface{} // Stub - LocalAI not available in standalone repo
}

func LoadSearchModel(modelPath string) (*SearchModel, error) {
	// LocalAI dependency removed - return stub model
	return &SearchModel{baseModel: nil}, nil
}

func (s *SearchModel) Close() error {
	return nil
}

func (s *SearchModel) Embed(ctx context.Context, text string) ([]float64, error) {
	// Placeholder: convert text to tokens and run forward pass
	_ = ctx
	_ = text
	return []float64{}, nil
}

func (s *SearchModel) Rerank(ctx context.Context, query string, documents []string) ([]float64, error) {
	_ = ctx
	_ = query
	_ = documents
	return make([]float64, len(documents)), nil
}
