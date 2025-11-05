package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// OpenSearchPersistence is the persistence layer for OpenSearch.
type OpenSearchPersistence struct {
	client   *http.Client
	url      string
	username string
	password string
	logger   *log.Logger
	index    string
}

// NewOpenSearchPersistence creates a new OpenSearch persistence layer.
func NewOpenSearchPersistence(url string, logger *log.Logger) (*OpenSearchPersistence, error) {
	// Remove trailing slash
	url = strings.TrimSuffix(url, "/")

	username := ""
	password := ""
	// TODO: Add authentication if needed

	p := &OpenSearchPersistence{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		url:      url,
		username: username,
		password: password,
		logger:   logger,
		index:    "embeddings",
	}

	// Ensure index exists
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := p.EnsureIndex(ctx); err != nil {
		return nil, fmt.Errorf("failed to ensure index: %w", err)
	}

	return p, nil
}

// EnsureIndex creates the embeddings index with vector mapping if it doesn't exist.
func (p *OpenSearchPersistence) EnsureIndex(ctx context.Context) error {
	// Check if index exists
	checkURL := fmt.Sprintf("%s/%s", p.url, p.index)
	req, err := http.NewRequestWithContext(ctx, "HEAD", checkURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	if p.username != "" && p.password != "" {
		req.SetBasicAuth(p.username, p.password)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to check index: %w", err)
	}
	resp.Body.Close()

	// Index exists
	if resp.StatusCode == 200 {
		return nil
	}

	// Create index with mapping
	mapping := map[string]any{
		"mappings": map[string]any{
			"properties": map[string]any{
				"vector": map[string]any{
					"type": "knn_vector",
					"dimension": 768,
					"method": map[string]any{
						"name": "hnsw",
						"space_type": "cosinesimil",
						"engine": "nmslib",
						"parameters": map[string]any{
							"ef_construction": 128,
							"m": 16,
						},
					},
				},
				"text": map[string]any{
					"type": "text",
				},
				"artifact_type": map[string]any{
					"type": "keyword",
				},
				"artifact_id": map[string]any{
					"type": "keyword",
				},
				"metadata": map[string]any{
					"type": "object",
				},
			},
		},
	}

	mappingJSON, err := json.Marshal(mapping)
	if err != nil {
		return fmt.Errorf("failed to marshal mapping: %w", err)
	}

	createURL := fmt.Sprintf("%s/%s", p.url, p.index)
	req, err = http.NewRequestWithContext(ctx, "PUT", createURL, bytes.NewReader(mappingJSON))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.username != "" && p.password != "" {
		req.SetBasicAuth(p.username, p.password)
	}

	resp, err = p.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 && resp.StatusCode != 201 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to create index: status %d, body: %s", resp.StatusCode, string(body))
	}

	return nil
}

// IndexVector indexes a vector in OpenSearch.
func (p *OpenSearchPersistence) IndexVector(key string, vector []float32, metadata map[string]any) error {
	ctx := context.Background()

	// Extract text and artifact info from metadata
	text := ""
	artifactType := ""
	artifactID := key
	if metadata != nil {
		if t, ok := metadata["label"].(string); ok {
			text = t
		}
		if at, ok := metadata["artifact_type"].(string); ok {
			artifactType = at
		}
		if aid, ok := metadata["artifact_id"].(string); ok {
			artifactID = aid
		}
	}

	// Convert vector to []float64 for JSON
	vectorFloat64 := make([]float64, len(vector))
	for i, v := range vector {
		vectorFloat64[i] = float64(v)
	}

	doc := map[string]any{
		"vector":       vectorFloat64,
		"text":         text,
		"artifact_type": artifactType,
		"artifact_id":  artifactID,
		"metadata":     metadata,
	}

	docJSON, err := json.Marshal(doc)
	if err != nil {
		return fmt.Errorf("failed to marshal document: %w", err)
	}

	// Index document
	indexURL := fmt.Sprintf("%s/%s/_doc/%s", p.url, p.index, key)
	req, err := http.NewRequestWithContext(ctx, "PUT", indexURL, bytes.NewReader(docJSON))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.username != "" && p.password != "" {
		req.SetBasicAuth(p.username, p.password)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to index vector: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 && resp.StatusCode != 201 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to index vector: status %d, body: %s", resp.StatusCode, string(body))
	}

	return nil
}

// SaveVector implements VectorPersistence interface by calling IndexVector.
func (p *OpenSearchPersistence) SaveVector(key string, vector []float32, metadata map[string]any) error {
	return p.IndexVector(key, vector, metadata)
}

// GetVector retrieves a vector from OpenSearch.
func (p *OpenSearchPersistence) GetVector(key string) ([]float32, map[string]any, error) {
	ctx := context.Background()

	getURL := fmt.Sprintf("%s/%s/_doc/%s", p.url, p.index, key)
	req, err := http.NewRequestWithContext(ctx, "GET", getURL, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}

	if p.username != "" && p.password != "" {
		req.SetBasicAuth(p.username, p.password)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get vector: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return nil, nil, fmt.Errorf("vector not found: %s", key)
	}

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, nil, fmt.Errorf("failed to get vector: status %d, body: %s", resp.StatusCode, string(body))
	}

	var docResp struct {
		Source struct {
			Vector       []float64           `json:"vector"`
			Text         string             `json:"text"`
			ArtifactType string             `json:"artifact_type"`
			ArtifactID   string             `json:"artifact_id"`
			Metadata     map[string]any     `json:"metadata"`
		} `json:"_source"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&docResp); err != nil {
		return nil, nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Convert []float64 to []float32
	vector := make([]float32, len(docResp.Source.Vector))
	for i, v := range docResp.Source.Vector {
		vector[i] = float32(v)
	}

	return vector, docResp.Source.Metadata, nil
}

// SearchSimilar implements VectorPersistence interface.
// Note: OpenSearch doesn't support threshold in the same way, so we filter results post-search.
func (p *OpenSearchPersistence) SearchSimilar(queryVector []float32, artifactType string, limit int, threshold float32) ([]VectorSearchResult, error) {
	results, err := p.searchSimilar(queryVector, "", artifactType, limit)
	if err != nil {
		return nil, err
	}

	// Filter by threshold
	filtered := []VectorSearchResult{}
	for _, result := range results {
		if result.Score >= threshold {
			filtered = append(filtered, result)
		}
	}

	return filtered, nil
}

// searchSimilar is the internal implementation (without threshold filtering).
func (p *OpenSearchPersistence) searchSimilar(queryVector []float32, query string, artifactType string, limit int) ([]VectorSearchResult, error) {
	ctx := context.Background()

	// Convert vector to []float64
	vectorFloat64 := make([]float64, len(queryVector))
	for i, v := range queryVector {
		vectorFloat64[i] = float64(v)
	}

	// Build query
	searchQuery := map[string]any{
		"size": limit,
	}

	// Add filters
	if artifactType != "" {
		searchQuery["query"] = map[string]any{
			"bool": map[string]any{
				"filter": []map[string]any{
					{
						"term": map[string]any{
							"artifact_type": artifactType,
						},
					},
				},
			},
		}
	}

	// Add hybrid search (vector + text)
	queries := []map[string]any{}

	// Vector similarity search
	queries = append(queries, map[string]any{
		"knn": map[string]any{
			"vector": map[string]any{
				"vector": vectorFloat64,
				"k":      limit,
			},
		},
	})

	// Text search (if query provided)
	if query != "" {
		queries = append(queries, map[string]any{
			"match": map[string]any{
				"text": map[string]any{
					"query": query,
				},
			},
		})
	}

	// Use hybrid query if both vector and text, otherwise use single query
	if len(queries) > 1 {
		// Hybrid query (OpenSearch 2.x supports this)
		if searchQuery["query"] == nil {
			searchQuery["query"] = map[string]any{
				"hybrid": map[string]any{
					"queries": queries,
				},
			}
		} else {
			// Combine with existing filter
			boolQuery := searchQuery["query"].(map[string]any)["bool"].(map[string]any)
			boolQuery["should"] = queries
			boolQuery["minimum_should_match"] = 1
		}
	} else if len(queries) == 1 {
		if searchQuery["query"] == nil {
			searchQuery["query"] = queries[0]
		} else {
			// Combine with existing filter
			boolQuery := searchQuery["query"].(map[string]any)["bool"].(map[string]any)
			boolQuery["should"] = []map[string]any{queries[0]}
			boolQuery["minimum_should_match"] = 1
		}
	}

	searchJSON, err := json.Marshal(searchQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal search query: %w", err)
	}

	// Execute search
	searchURL := fmt.Sprintf("%s/%s/_search", p.url, p.index)
	req, err := http.NewRequestWithContext(ctx, "POST", searchURL, bytes.NewReader(searchJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.username != "" && p.password != "" {
		req.SetBasicAuth(p.username, p.password)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to search: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("failed to search: status %d, body: %s", resp.StatusCode, string(body))
	}

	var searchResp struct {
		Hits struct {
			Hits []struct {
				ID     string `json:"_id"`
				Score  float64 `json:"_score"`
				Source struct {
					Vector       []float64           `json:"vector"`
					Text         string             `json:"text"`
					ArtifactType string             `json:"artifact_type"`
					ArtifactID   string             `json:"artifact_id"`
					Metadata     map[string]any     `json:"metadata"`
				} `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("failed to decode search response: %w", err)
	}

	results := make([]VectorSearchResult, len(searchResp.Hits.Hits))
	for i, hit := range searchResp.Hits.Hits {
		// Convert []float64 back to []float32
		vector := make([]float32, len(hit.Source.Vector))
		for j, v := range hit.Source.Vector {
			vector[j] = float32(v)
		}

		results[i] = VectorSearchResult{
			Key:          hit.ID,
			ArtifactType: hit.Source.ArtifactType,
			ArtifactID:   hit.Source.ArtifactID,
			Vector:       vector,
			Metadata:     hit.Source.Metadata,
			Score:        float32(hit.Score),
			Text:         hit.Source.Text,
		}
	}

	return results, nil
}

// SearchByText performs text-based search in OpenSearch.
func (p *OpenSearchPersistence) SearchByText(query string, artifactType string, limit int) ([]VectorSearchResult, error) {
	ctx := context.Background()

	searchQuery := map[string]any{
		"size": limit,
		"query": map[string]any{
			"bool": map[string]any{
				"must": []map[string]any{
					{
						"match": map[string]any{
							"text": map[string]any{
								"query": query,
							},
						},
					},
				},
			},
		},
	}

	if artifactType != "" {
		boolQuery := searchQuery["query"].(map[string]any)["bool"].(map[string]any)
		boolQuery["filter"] = []map[string]any{
			{
				"term": map[string]any{
					"artifact_type": artifactType,
				},
			},
		}
	}

	searchJSON, err := json.Marshal(searchQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal search query: %w", err)
	}

	searchURL := fmt.Sprintf("%s/%s/_search", p.url, p.index)
	req, err := http.NewRequestWithContext(ctx, "POST", searchURL, bytes.NewReader(searchJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.username != "" && p.password != "" {
		req.SetBasicAuth(p.username, p.password)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to search: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("failed to search: status %d, body: %s", resp.StatusCode, string(body))
	}

	var searchResp struct {
		Hits struct {
			Hits []struct {
				ID     string `json:"_id"`
				Score  float64 `json:"_score"`
				Source struct {
					Vector       []float64           `json:"vector"`
					Text         string             `json:"text"`
					ArtifactType string             `json:"artifact_type"`
					ArtifactID   string             `json:"artifact_id"`
					Metadata     map[string]any     `json:"metadata"`
				} `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("failed to decode search response: %w", err)
	}

	results := make([]VectorSearchResult, len(searchResp.Hits.Hits))
	for i, hit := range searchResp.Hits.Hits {
		// Convert []float64 back to []float32
		vector := make([]float32, len(hit.Source.Vector))
		for j, v := range hit.Source.Vector {
			vector[j] = float32(v)
		}

		results[i] = VectorSearchResult{
			Key:          hit.ID,
			ArtifactType: hit.Source.ArtifactType,
			ArtifactID:   hit.Source.ArtifactID,
			Vector:       vector,
			Metadata:     hit.Source.Metadata,
			Score:        float32(hit.Score),
			Text:         hit.Source.Text,
		}
	}

	return results, nil
}

