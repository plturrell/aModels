package embeddings

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove"
)

// GloVeSearch provides GloVe-based semantic search capabilities.
type GloVeSearch struct {
	model      *glove.Model
	dimension  int
	indexCache map[string][]float32
	cacheMu    sync.RWMutex
}

// NewGloVeSearch creates a new GloVe search engine.
func NewGloVeSearch(model *glove.Model, dimension int) *GloVeSearch {
	return &GloVeSearch{
		model:      model,
		dimension:  dimension,
		indexCache: make(map[string][]float32),
	}
}

// SearchResult represents a search result with similarity score.
type SearchResult struct {
	Document   string
	Score      float32
	Embedding  []float32
	Highlights []string
}

// SemanticSearch performs semantic search using GloVe embeddings.
func (g *GloVeSearch) SemanticSearch(ctx context.Context, query string, documents []string, topK int) ([]SearchResult, error) {
	// Get query embedding
	queryEmb, err := g.GetEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get query embedding: %w", err)
	}

	// Calculate similarities
	results := make([]SearchResult, 0, len(documents))
	for _, doc := range documents {
		docEmb, err := g.GetEmbedding(ctx, doc)
		if err != nil {
			continue
		}

		score := cosineSimilarity(queryEmb, docEmb)
		highlights := extractHighlights(query, doc, 3)

		results = append(results, SearchResult{
			Document:   doc,
			Score:      score,
			Embedding:  docEmb,
			Highlights: highlights,
		})
	}

	// Sort by score
	sortByScore(results)

	// Return top K
	if topK > 0 && topK < len(results) {
		results = results[:topK]
	}

	return results, nil
}

// GetEmbedding retrieves or computes embedding for text.
func (g *GloVeSearch) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	g.cacheMu.RLock()
	if cached, ok := g.indexCache[text]; ok {
		g.cacheMu.RUnlock()
		return cached, nil
	}
	g.cacheMu.RUnlock()

	// Tokenize and average word vectors
	words := tokenize(text)
	if len(words) == 0 {
		return make([]float32, g.dimension), nil
	}

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

	// Average
	if count > 0 {
		for i := range embedding {
			embedding[i] /= float32(count)
		}
	}

	// Cache result
	g.cacheMu.Lock()
	g.indexCache[text] = embedding
	g.cacheMu.Unlock()

	return embedding, nil
}

// FindSimilarDocuments finds documents similar to the query.
func (g *GloVeSearch) FindSimilarDocuments(ctx context.Context, query string, corpus []string, threshold float32) ([]string, error) {
	results, err := g.SemanticSearch(ctx, query, corpus, 0)
	if err != nil {
		return nil, err
	}

	similar := make([]string, 0)
	for _, result := range results {
		if result.Score >= threshold {
			similar = append(similar, result.Document)
		}
	}

	return similar, nil
}

// ClusterDocuments clusters documents using GloVe embeddings.
func (g *GloVeSearch) ClusterDocuments(ctx context.Context, documents []string, numClusters int) ([][]string, error) {
	if numClusters <= 0 || len(documents) == 0 {
		return nil, fmt.Errorf("invalid parameters")
	}

	// Get embeddings for all documents
	embeddings := make([][]float32, len(documents))
	for i, doc := range documents {
		emb, err := g.GetEmbedding(ctx, doc)
		if err != nil {
			return nil, err
		}
		embeddings[i] = emb
	}

	// Simple k-means clustering
	clusters := kMeansClustering(embeddings, numClusters, 10)

	// Group documents by cluster
	result := make([][]string, numClusters)
	for i, clusterID := range clusters {
		result[clusterID] = append(result[clusterID], documents[i])
	}

	return result, nil
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

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func extractHighlights(query, document string, maxHighlights int) []string {
	queryWords := tokenize(query)
	docWords := tokenize(document)
	
	highlights := make([]string, 0, maxHighlights)
	for _, qw := range queryWords {
		for i, dw := range docWords {
			if strings.Contains(dw, qw) {
				start := max(0, i-2)
				end := min(len(docWords), i+3)
				highlight := strings.Join(docWords[start:end], " ")
				highlights = append(highlights, highlight)
				if len(highlights) >= maxHighlights {
					return highlights
				}
			}
		}
	}
	return highlights
}

func sortByScore(results []SearchResult) {
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
}

func kMeansClustering(embeddings [][]float32, k int, maxIter int) []int {
	n := len(embeddings)
	if n == 0 || k <= 0 {
		return nil
	}

	// Initialize centroids randomly
	centroids := make([][]float32, k)
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, len(embeddings[0]))
		copy(centroids[i], embeddings[i%n])
	}

	assignments := make([]int, n)

	for iter := 0; iter < maxIter; iter++ {
		// Assign points to nearest centroid
		for i, emb := range embeddings {
			minDist := float32(math.MaxFloat32)
			for j, centroid := range centroids {
				dist := euclideanDistance(emb, centroid)
				if dist < minDist {
					minDist = dist
					assignments[i] = j
				}
			}
		}

		// Update centroids
		counts := make([]int, k)
		newCentroids := make([][]float32, k)
		for i := range newCentroids {
			newCentroids[i] = make([]float32, len(embeddings[0]))
		}

		for i, emb := range embeddings {
			cluster := assignments[i]
			counts[cluster]++
			for j, v := range emb {
				newCentroids[cluster][j] += v
			}
		}

		for i := range newCentroids {
			if counts[i] > 0 {
				for j := range newCentroids[i] {
					newCentroids[i][j] /= float32(counts[i])
				}
			}
		}

		centroids = newCentroids
	}

	return assignments
}

func euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
