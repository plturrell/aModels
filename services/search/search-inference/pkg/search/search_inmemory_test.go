package search

import "testing"

func TestSearchInMemoryOrdersBySimilarity(t *testing.T) {
	service := &SearchService{
		useInMemory: true,
		memDocs: map[string]*storedDocument{
			"doc-1": {
				Content:   "alpha",
				Metadata:  map[string]interface{}{"domain": "hr"},
				Embedding: []float64{1, 0},
			},
			"doc-2": {
				Content:   "beta",
				Metadata:  map[string]interface{}{"domain": "it"},
				Embedding: []float64{0.6, 0.8},
			},
			"doc-3": {
				Content:   "gamma",
				Metadata:  map[string]interface{}{"domain": "finance"},
				Embedding: []float64{0, 1},
			},
		},
	}

	queryEmbedding := []float64{1, 0}
	results := service.searchInMemory(queryEmbedding, 0, nil)

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	if results[0].ID != "doc-1" {
		t.Fatalf("expected most similar document to be doc-1, got %s", results[0].ID)
	}

	if results[0].Score != results[0].Similarity {
		t.Fatalf("expected score to equal similarity, got %f vs %f", results[0].Score, results[0].Similarity)
	}

	top1 := service.searchInMemory(queryEmbedding, 1, nil)
	if len(top1) != 1 || top1[0].ID != "doc-1" {
		t.Fatalf("expected top1 to return doc-1, got %+v", top1)
	}

	filtered := service.searchInMemory(queryEmbedding, 0, map[string]string{"metadata.domain": "hr"})
	if len(filtered) != 1 || filtered[0].ID != "doc-1" {
		t.Fatalf("expected filter to keep only doc-1, got %+v", filtered)
	}
}

func TestMetadataMatchesFilters(t *testing.T) {
	metadata := map[string]interface{}{
		"domain":  "finance",
		"task":    "policy",
		"owner":   "hr",
		"nested":  map[string]interface{}{"field": "value"},
		"number":  42,
		"boolean": true,
	}

	tests := []struct {
		name    string
		filters map[string]string
		want    bool
	}{
		{"empty filters", nil, true},
		{"domain match", map[string]string{"metadata.domain": "finance"}, true},
		{"domain mismatch", map[string]string{"metadata.domain": "hr"}, false},
		{"non prefixed key", map[string]string{"owner": "hr"}, true},
		{"non prefixed mismatch", map[string]string{"owner": "legal"}, false},
		{"number coerced", map[string]string{"metadata.number": "42"}, true},
		{"bool coerced", map[string]string{"metadata.boolean": "true"}, true},
		{"nested missing", map[string]string{"metadata.nested": "value"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := metadataMatchesFilters(metadata, tt.filters); got != tt.want {
				t.Fatalf("metadataMatchesFilters() = %v, want %v", got, tt.want)
			}
		})
	}
}
