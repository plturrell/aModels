package embeddings

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
)

// VectorProvider describes the minimal interface required from a GloVe model implementation.
type VectorProvider interface {
	GetVector(ctx context.Context, word string) ([]float32, error)
}

// GloVeLocalAI provides GloVe embeddings for local AI inference.
type GloVeLocalAI struct {
	model     VectorProvider
	dimension int
	cache     map[string][]float32
	cacheMu   sync.RWMutex
}

// NewGloVeLocalAI creates a new GloVe embedding provider for LocalAI.
func NewGloVeLocalAI(model VectorProvider, dimension int) *GloVeLocalAI {
	return &GloVeLocalAI{
		model:     model,
		dimension: dimension,
		cache:     make(map[string][]float32),
	}
}

// EmbeddingRequest represents a request for embeddings.
type EmbeddingRequest struct {
	Input      []string
	Model      string
	Dimensions int
}

// EmbeddingResponse represents the embedding response.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingUsage  `json:"usage"`
}

// EmbeddingData represents a single embedding.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingUsage represents token usage.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// CreateEmbeddings generates embeddings for the given inputs.
func (g *GloVeLocalAI) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	data := make([]EmbeddingData, len(req.Input))
	totalTokens := 0

	for i, text := range req.Input {
		embedding, err := g.GetEmbedding(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to get embedding for input %d: %w", i, err)
		}

		tokens := len(strings.Fields(text))
		totalTokens += tokens

		data[i] = EmbeddingData{
			Object:    "embedding",
			Embedding: embedding,
			Index:     i,
		}
	}

	return &EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  req.Model,
		Usage: EmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}, nil
}

// GetEmbedding retrieves or computes embedding for text.
func (g *GloVeLocalAI) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	g.cacheMu.RLock()
	if cached, ok := g.cache[text]; ok {
		g.cacheMu.RUnlock()
		return cached, nil
	}
	g.cacheMu.RUnlock()

	// Tokenize text
	words := tokenize(text)
	if len(words) == 0 {
		return make([]float32, g.dimension), nil
	}

	// Average word vectors
	embedding := make([]float32, g.dimension)
	count := 0

	for _, word := range words {
		vec, err := g.model.GetVector(ctx, word)
		if err != nil {
			continue
		}
		for i, v := range vec {
			if i < g.dimension {
				embedding[i] += v
			}
		}
		count++
	}

	// Average and normalize
	if count > 0 {
		for i := range embedding {
			embedding[i] /= float32(count)
		}
		embedding = normalize(embedding)
	}

	// Cache result
	g.cacheMu.Lock()
	g.cache[text] = embedding
	g.cacheMu.Unlock()

	return embedding, nil
}

// ComputeSimilarity computes cosine similarity between two texts.
func (g *GloVeLocalAI) ComputeSimilarity(ctx context.Context, text1, text2 string) (float32, error) {
	emb1, err := g.GetEmbedding(ctx, text1)
	if err != nil {
		return 0, err
	}

	emb2, err := g.GetEmbedding(ctx, text2)
	if err != nil {
		return 0, err
	}

	return cosineSimilarity(emb1, emb2), nil
}

// FindMostSimilar finds the most similar text from a list.
func (g *GloVeLocalAI) FindMostSimilar(ctx context.Context, query string, candidates []string) (string, float32, error) {
	queryEmb, err := g.GetEmbedding(ctx, query)
	if err != nil {
		return "", 0, err
	}

	maxSim := float32(-1)
	bestMatch := ""

	for _, candidate := range candidates {
		candEmb, err := g.GetEmbedding(ctx, candidate)
		if err != nil {
			continue
		}

		sim := cosineSimilarity(queryEmb, candEmb)
		if sim > maxSim {
			maxSim = sim
			bestMatch = candidate
		}
	}

	return bestMatch, maxSim, nil
}

// BatchGetEmbeddings gets embeddings for multiple texts efficiently.
func (g *GloVeLocalAI) BatchGetEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))

	var wg sync.WaitGroup
	errors := make(chan error, len(texts))

	for i, text := range texts {
		wg.Add(1)
		go func(idx int, txt string) {
			defer wg.Done()
			emb, err := g.GetEmbedding(ctx, txt)
			if err != nil {
				errors <- err
				return
			}
			embeddings[idx] = emb
		}(i, text)
	}

	wg.Wait()
	close(errors)

	if len(errors) > 0 {
		return nil, <-errors
	}

	return embeddings, nil
}

// ClearCache clears the embedding cache.
func (g *GloVeLocalAI) ClearCache() {
	g.cacheMu.Lock()
	g.cache = make(map[string][]float32)
	g.cacheMu.Unlock()
}

// Helper functions

func tokenize(text string) []string {
	text = strings.ToLower(text)
	words := strings.Fields(text)
	result := make([]string, 0, len(words))
	for _, word := range words {
		word = strings.Trim(word, ".,!?;:\"'()[]{}")
		if len(word) > 0 {
			result = append(result, word)
		}
	}
	return result
}

func normalize(vec []float32) []float32 {
	norm := 0.0
	for _, v := range vec {
		norm += float64(v * v)
	}
	if norm == 0 {
		result := make([]float32, len(vec))
		copy(result, vec)
		return result
	}

	scale := float32(1.0 / math.Sqrt(norm))
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = v * scale
	}
	return result
}

func cosineSimilarity(a, b []float32) float32 {
    if len(a) != len(b) || len(a) == 0 {
        return 0.0
    }
    var dot, na, nb float64
    for i := range a {
        av := float64(a[i])
        bv := float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    }
    if na == 0 || nb == 0 {
        return 0.0
    }
    return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
