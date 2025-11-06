package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// EmbeddingService generates embeddings for text content
type EmbeddingService struct {
	serviceURL string
	httpClient *http.Client
	logger     *log.Logger
}

// NewEmbeddingService creates a new embedding service
func NewEmbeddingService(serviceURL string, logger *log.Logger) *EmbeddingService {
	if serviceURL == "" {
		serviceURL = "http://localhost:8081" // Default LocalAI URL
	}
	return &EmbeddingService{
		serviceURL: serviceURL,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		logger:     logger,
	}
}

// GenerateEmbedding generates an embedding vector for text
func (es *EmbeddingService) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	if es.serviceURL == "" {
		return nil, fmt.Errorf("embedding service URL not configured")
	}

	// Call LocalAI embedding endpoint
	url := fmt.Sprintf("%s/v1/embeddings", es.serviceURL)
	
	requestBody := map[string]interface{}{
		"input": text,
		"model": "text-embedding-ada-002", // or local embedding model
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := es.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call embedding service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding service returned status %d: %s", resp.StatusCode, string(body))
	}

	var embeddingResponse struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&embeddingResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(embeddingResponse.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// Convert float64 to float32
	embedding := make([]float32, len(embeddingResponse.Data[0].Embedding))
	for i, v := range embeddingResponse.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	if es.logger != nil {
		es.logger.Printf("Generated embedding: dimension=%d", len(embedding))
	}

	return embedding, nil
}

// GenerateEmbeddingsBatch generates embeddings for multiple texts
func (es *EmbeddingService) GenerateEmbeddingsBatch(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	
	for i, text := range texts {
		embedding, err := es.GenerateEmbedding(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding for text %d: %w", i, err)
		}
		embeddings[i] = embedding
	}

	return embeddings, nil
}

