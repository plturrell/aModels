package search

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLocalAIEmbedderEmbed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2,0.3],"index":0}],"model":"test-model"}`))
	}))
	defer server.Close()

	embedder := NewLocalAIEmbedder(server.URL, "test-key")

	embedding, err := embedder.Embed(context.Background(), "search text")
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}

	expected := []float64{0.1, 0.2, 0.3}
	if len(embedding) != len(expected) {
		t.Fatalf("expected embedding length %d, got %d", len(expected), len(embedding))
	}
	for i := range expected {
		if embedding[i] != expected[i] {
			t.Fatalf("expected embedding[%d]=%f, got %f", i, expected[i], embedding[i])
		}
	}
}

func TestLocalAIEmbedderEmbedBatchError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("{}"))
	}))
	defer server.Close()

	embedder := NewLocalAIEmbedder(server.URL, "")

	if _, err := embedder.EmbedBatch(context.Background(), []string{"a", "b"}); err == nil {
		t.Fatalf("expected error from EmbedBatch when server returns non-200 status")
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		b    []float64
		want float64
	}{
		{
			name: "identical vectors",
			a:    []float64{1, 2, 3},
			b:    []float64{1, 2, 3},
			want: 1,
		},
		{
			name: "orthogonal vectors",
			a:    []float64{1, 0},
			b:    []float64{0, 1},
			want: 0,
		},
		{
			name: "different magnitudes",
			a:    []float64{1, 1},
			b:    []float64{2, 2},
			want: 1,
		},
		{
			name: "mismatched lengths",
			a:    []float64{1, 2},
			b:    []float64{1},
			want: 0,
		},
		{
			name: "zero vector",
			a:    []float64{0, 0},
			b:    []float64{1, 2},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cosineSimilarity(tt.a, tt.b)
			if !almostEqual(got, tt.want) {
				t.Fatalf("cosineSimilarity() = %f, want %f", got, tt.want)
			}
		})
	}
}

func almostEqual(a, b float64) bool {
	const epsilon = 1e-9
	if a > b {
		return a-b < epsilon
	}
	return b-a < epsilon
}
