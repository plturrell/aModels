package embeddings

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
)

type stubVectorProvider struct {
	mu      sync.Mutex
	vectors map[string][]float32
	calls   map[string]int
}

func newStubVectorProvider(vectors map[string][]float32) *stubVectorProvider {
	return &stubVectorProvider{
		vectors: vectors,
		calls:   make(map[string]int),
	}
}

func (s *stubVectorProvider) GetVector(ctx context.Context, word string) ([]float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls[word]++
	if vec, ok := s.vectors[word]; ok {
		return vec, nil
	}
	return nil, fmt.Errorf("vector not found: %s", word)
}

func (s *stubVectorProvider) callCount(word string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls[word]
}

func almostEqual(a, b float32) bool {
	const epsilon = 1e-4
	return float32(math.Abs(float64(a-b))) <= epsilon
}

func TestCreateEmbeddings(t *testing.T) {
	provider := newStubVectorProvider(map[string][]float32{
		"hello": {1, 0, 0},
		"world": {0, 1, 0},
	})

	g := NewGloVeLocalAI(provider, 3)
	req := EmbeddingRequest{
		Input: []string{"hello world", "hello"},
		Model: "test-model",
	}

	resp, err := g.CreateEmbeddings(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %v", err)
	}

	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 embedding results, got %d", len(resp.Data))
	}

	if resp.Usage.TotalTokens != 3 {
		t.Fatalf("expected usage total tokens 3, got %d", resp.Usage.TotalTokens)
	}

	emb := resp.Data[0].Embedding
	if len(emb) != 3 {
		t.Fatalf("expected embedding length 3, got %d", len(emb))
	}

	// After normalization the vector should roughly point along (1, 1, 0)
	if !almostEqual(emb[0], 0.70710677) || !almostEqual(emb[1], 0.70710677) {
		t.Fatalf("embedding not normalized as expected: %v", emb)
	}
}

func TestGetEmbeddingCachesResults(t *testing.T) {
	provider := newStubVectorProvider(map[string][]float32{
		"cache": {0.5, 0.5, 0.5},
	})

	g := NewGloVeLocalAI(provider, 3)
	_, err := g.GetEmbedding(context.Background(), "cache")
	if err != nil {
		t.Fatalf("first GetEmbedding failed: %v", err)
	}

	if provider.callCount("cache") != 1 {
		t.Fatalf("expected 1 call to provider, got %d", provider.callCount("cache"))
	}

	// Second call should use cache
	_, err = g.GetEmbedding(context.Background(), "cache")
	if err != nil {
		t.Fatalf("second GetEmbedding failed: %v", err)
	}

	if provider.callCount("cache") != 1 {
		t.Fatalf("expected cached result, provider call count %d", provider.callCount("cache"))
	}
}

func TestComputeSimilarity(t *testing.T) {
	provider := newStubVectorProvider(map[string][]float32{
		"foo": {1, 0},
		"bar": {0, 1},
		"baz": {1, 0},
	})

	g := NewGloVeLocalAI(provider, 2)

	sim, err := g.ComputeSimilarity(context.Background(), "foo", "baz")
	if err != nil {
		t.Fatalf("ComputeSimilarity returned error: %v", err)
	}

	if !almostEqual(float32(sim), 1) {
		t.Fatalf("expected similarity 1, got %f", sim)
	}

	sim, err = g.ComputeSimilarity(context.Background(), "foo", "bar")
	if err != nil {
		t.Fatalf("ComputeSimilarity returned error: %v", err)
	}

	if !almostEqual(float32(sim), 0) {
		t.Fatalf("expected similarity near 0, got %f", sim)
	}
}

func TestFindMostSimilar(t *testing.T) {
	provider := newStubVectorProvider(map[string][]float32{
		"query":  {1, 0},
		"first":  {0.9, 0.1},
		"second": {0, 1},
	})

	g := NewGloVeLocalAI(provider, 2)
	match, sim, err := g.FindMostSimilar(context.Background(), "query", []string{"first", "second"})
	if err != nil {
		t.Fatalf("FindMostSimilar returned error: %v", err)
	}

	if match != "first" {
		t.Fatalf("expected 'first' to be most similar, got %s", match)
	}

	if sim <= 0 {
		t.Fatalf("expected positive similarity score, got %f", sim)
	}
}

func TestBatchGetEmbeddings(t *testing.T) {
	provider := newStubVectorProvider(map[string][]float32{
		"a": {1, 0},
		"b": {0, 1},
	})

	g := NewGloVeLocalAI(provider, 2)
	embeddings, err := g.BatchGetEmbeddings(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("BatchGetEmbeddings returned error: %v", err)
	}

	if len(embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(embeddings))
	}

	if provider.callCount("a") != 1 || provider.callCount("b") != 1 {
		t.Fatalf("unexpected provider call counts: a=%d b=%d", provider.callCount("a"), provider.callCount("b"))
	}
}

func TestClearCache(t *testing.T) {
	provider := newStubVectorProvider(map[string][]float32{
		"reset": {1, 1, 1},
	})

	g := NewGloVeLocalAI(provider, 3)
	if _, err := g.GetEmbedding(context.Background(), "reset"); err != nil {
		t.Fatalf("GetEmbedding failed: %v", err)
	}

	g.ClearCache()

	if _, err := g.GetEmbedding(context.Background(), "reset"); err != nil {
		t.Fatalf("GetEmbedding after ClearCache failed: %v", err)
	}

	if provider.callCount("reset") != 2 {
		t.Fatalf("expected provider to be called again after ClearCache, got %d calls", provider.callCount("reset"))
	}
}
