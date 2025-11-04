package elasticsearch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/schema"
)

// ElasticsearchVectorStore provides vector storage and retrieval using Elasticsearch
type ElasticsearchVectorStore struct {
	client       *http.Client
	baseURL      string
	indexName    string
	username     string
	password     string
	embeddingDim int
}

// Document represents a document in the vector store
type Document struct {
	ID        string                 `json:"id"`
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata"`
	Embedding []float64              `json:"embedding"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// SearchResult represents a search result
type SearchResult struct {
	Document   *Document `json:"document"`
	Score      float64   `json:"score"`
	Similarity float64   `json:"similarity"`
}

// SearchRequest represents a search request
type SearchRequest struct {
	Query               string                 `json:"query"`
	Embedding           []float64              `json:"embedding,omitempty"`
	TopK                int                    `json:"top_k,omitempty"`
	Filter              map[string]interface{} `json:"filter,omitempty"`
	SimilarityThreshold float64                `json:"similarity_threshold,omitempty"`
}

// NewElasticsearchVectorStore creates a new Elasticsearch vector store
func NewElasticsearchVectorStore(baseURL, indexName, username, password string, embeddingDim int) *ElasticsearchVectorStore {
	return &ElasticsearchVectorStore{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		baseURL:      baseURL,
		indexName:    indexName,
		username:     username,
		password:     password,
		embeddingDim: embeddingDim,
	}
}

// Initialize creates the Elasticsearch index with proper mapping
func (es *ElasticsearchVectorStore) Initialize(ctx context.Context) error {
	// Check if index exists
	exists, err := es.indexExists(ctx)
	if err != nil {
		return fmt.Errorf("failed to check index existence: %w", err)
	}

	if exists {
		return nil // Index already exists
	}

	// Create index with mapping
	mapping := map[string]interface{}{
		"mappings": map[string]interface{}{
			"properties": map[string]interface{}{
				"id": map[string]interface{}{
					"type": "keyword",
				},
				"content": map[string]interface{}{
					"type":     "text",
					"analyzer": "standard",
				},
				"metadata": map[string]interface{}{
					"type": "object",
				},
				"embedding": map[string]interface{}{
					"type":       "dense_vector",
					"dims":       es.embeddingDim,
					"index":      true,
					"similarity": "cosine",
				},
				"created_at": map[string]interface{}{
					"type": "date",
				},
				"updated_at": map[string]interface{}{
					"type": "date",
				},
			},
		},
		"settings": map[string]interface{}{
			"number_of_shards":   1,
			"number_of_replicas": 0,
		},
	}

	return es.createIndex(ctx, mapping)
}

// AddDocuments adds documents to the vector store
func (es *ElasticsearchVectorStore) AddDocuments(ctx context.Context, documents []schema.Document) error {
	if len(documents) == 0 {
		return nil
	}

	// Prepare bulk request
	var bulkBody strings.Builder
	for _, doc := range documents {
		// Convert schema.Document to our Document format
		esDoc := &Document{
			ID:        doc.ID,
			Content:   doc.PageContent,
			Metadata:  doc.Metadata,
			Embedding: doc.Embedding,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}

		// Add index action
		indexAction := map[string]interface{}{
			"index": map[string]interface{}{
				"_index": es.indexName,
				"_id":    esDoc.ID,
			},
		}

		indexActionJSON, _ := json.Marshal(indexAction)
		bulkBody.WriteString(string(indexActionJSON))
		bulkBody.WriteString("\n")

		// Add document
		docJSON, _ := json.Marshal(esDoc)
		bulkBody.WriteString(string(docJSON))
		bulkBody.WriteString("\n")
	}

	// Execute bulk request
	return es.bulkRequest(ctx, bulkBody.String())
}

// SimilaritySearch performs similarity search using vector embeddings
func (es *ElasticsearchVectorStore) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
	// This is a placeholder - in a real implementation, you would:
	// 1. Generate embeddings for the query using LocalAI
	// 2. Perform vector similarity search in Elasticsearch
	// 3. Return the results as schema.Document

	searchReq := SearchRequest{
		Query: query,
		TopK:  k,
	}

	results, err := es.search(ctx, searchReq)
	if err != nil {
		return nil, err
	}

	// Convert results to schema.Document
	documents := make([]schema.Document, len(results))
	for i, result := range results {
		documents[i] = schema.Document{
			ID:          result.Document.ID,
			PageContent: result.Document.Content,
			Metadata:    result.Document.Metadata,
			Embedding:   result.Document.Embedding,
		}
	}

	return documents, nil
}

// SimilaritySearchWithScore performs similarity search and returns scores
func (es *ElasticsearchVectorStore) SimilaritySearchWithScore(ctx context.Context, query string, k int) ([]schema.Document, []float64, error) {
	searchReq := SearchRequest{
		Query: query,
		TopK:  k,
	}

	results, err := es.search(ctx, searchReq)
	if err != nil {
		return nil, nil, err
	}

	// Convert results to schema.Document
	documents := make([]schema.Document, len(results))
	scores := make([]float64, len(results))
	for i, result := range results {
		documents[i] = schema.Document{
			ID:          result.Document.ID,
			PageContent: result.Document.Content,
			Metadata:    result.Document.Metadata,
			Embedding:   result.Document.Embedding,
		}
		scores[i] = result.Score
	}

	return documents, scores, nil
}

// DeleteDocuments deletes documents by IDs
func (es *ElasticsearchVectorStore) DeleteDocuments(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	// Prepare bulk delete request
	var bulkBody strings.Builder
	for _, id := range ids {
		deleteAction := map[string]interface{}{
			"delete": map[string]interface{}{
				"_index": es.indexName,
				"_id":    id,
			},
		}

		deleteActionJSON, _ := json.Marshal(deleteAction)
		bulkBody.WriteString(string(deleteActionJSON))
		bulkBody.WriteString("\n")
	}

	return es.bulkRequest(ctx, bulkBody.String())
}

// GetDocument retrieves a document by ID
func (es *ElasticsearchVectorStore) GetDocument(ctx context.Context, id string) (*schema.Document, error) {
	url := fmt.Sprintf("%s/%s/_doc/%s", es.baseURL, es.indexName, id)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	es.setAuth(req)

	resp, err := es.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("document not found")
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("elasticsearch error: %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	source, ok := result["_source"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid document format")
	}

	// Convert to schema.Document
	doc := &schema.Document{
		ID: id,
	}

	if content, ok := source["content"].(string); ok {
		doc.PageContent = content
	}

	if metadata, ok := source["metadata"].(map[string]interface{}); ok {
		doc.Metadata = metadata
	}

	if embedding, ok := source["embedding"].([]interface{}); ok {
		doc.Embedding = make([]float64, len(embedding))
		for i, v := range embedding {
			if f, ok := v.(float64); ok {
				doc.Embedding[i] = f
			}
		}
	}

	return doc, nil
}

// UpdateDocument updates a document
func (es *ElasticsearchVectorStore) UpdateDocument(ctx context.Context, id string, document *schema.Document) error {
	esDoc := &Document{
		ID:        id,
		Content:   document.PageContent,
		Metadata:  document.Metadata,
		Embedding: document.Embedding,
		UpdatedAt: time.Now(),
	}

	docJSON, err := json.Marshal(esDoc)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("%s/%s/_doc/%s", es.baseURL, es.indexName, id)
	req, err := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewBuffer(docJSON))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	es.setAuth(req)

	resp, err := es.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("elasticsearch error: %d", resp.StatusCode)
	}

	return nil
}

// GetStats returns statistics about the vector store
func (es *ElasticsearchVectorStore) GetStats(ctx context.Context) (map[string]interface{}, error) {
	url := fmt.Sprintf("%s/%s/_stats", es.baseURL, es.indexName)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	es.setAuth(req)

	resp, err := es.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("elasticsearch error: %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

// Helper methods

func (es *ElasticsearchVectorStore) indexExists(ctx context.Context) (bool, error) {
	url := fmt.Sprintf("%s/%s", es.baseURL, es.indexName)

	req, err := http.NewRequestWithContext(ctx, "HEAD", url, nil)
	if err != nil {
		return false, err
	}

	es.setAuth(req)

	resp, err := es.client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK, nil
}

func (es *ElasticsearchVectorStore) createIndex(ctx context.Context, mapping map[string]interface{}) error {
	url := fmt.Sprintf("%s/%s", es.baseURL, es.indexName)

	mappingJSON, err := json.Marshal(mapping)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewBuffer(mappingJSON))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	es.setAuth(req)

	resp, err := es.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("elasticsearch error: %d", resp.StatusCode)
	}

	return nil
}

func (es *ElasticsearchVectorStore) search(ctx context.Context, searchReq SearchRequest) ([]SearchResult, error) {
	// This is a simplified search implementation
	// In a real implementation, you would construct a proper Elasticsearch query
	// with vector similarity search using the kNN query

	query := map[string]interface{}{
		"query": map[string]interface{}{
			"match": map[string]interface{}{
				"content": searchReq.Query,
			},
		},
		"size": searchReq.TopK,
	}

	queryJSON, err := json.Marshal(query)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/%s/_search", es.baseURL, es.indexName)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(queryJSON))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	es.setAuth(req)

	resp, err := es.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("elasticsearch error: %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	// Parse search results
	hits, ok := result["hits"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid search result format")
	}

	hitsList, ok := hits["hits"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid hits format")
	}

	results := make([]SearchResult, len(hitsList))
	for i, hit := range hitsList {
		hitMap, ok := hit.(map[string]interface{})
		if !ok {
			continue
		}

		source, ok := hitMap["_source"].(map[string]interface{})
		if !ok {
			continue
		}

		score, _ := hitMap["_score"].(float64)

		// Convert to Document
		doc := &Document{}
		if id, ok := source["id"].(string); ok {
			doc.ID = id
		}
		if content, ok := source["content"].(string); ok {
			doc.Content = content
		}
		if metadata, ok := source["metadata"].(map[string]interface{}); ok {
			doc.Metadata = metadata
		}
		if embedding, ok := source["embedding"].([]interface{}); ok {
			doc.Embedding = make([]float64, len(embedding))
			for j, v := range embedding {
				if f, ok := v.(float64); ok {
					doc.Embedding[j] = f
				}
			}
		}

		results[i] = SearchResult{
			Document:   doc,
			Score:      score,
			Similarity: score, // In a real implementation, calculate actual similarity
		}
	}

	return results, nil
}

func (es *ElasticsearchVectorStore) bulkRequest(ctx context.Context, body string) error {
	url := fmt.Sprintf("%s/_bulk", es.baseURL)

	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(body))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/x-ndjson")
	es.setAuth(req)

	resp, err := es.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("elasticsearch bulk error: %d", resp.StatusCode)
	}

	return nil
}

func (es *ElasticsearchVectorStore) setAuth(req *http.Request) {
	if es.username != "" && es.password != "" {
		req.SetBasicAuth(es.username, es.password)
	}
}
