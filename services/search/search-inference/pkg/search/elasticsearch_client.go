package search

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"time"

	elasticsearch "github.com/elastic/go-elasticsearch/v8"
)

// ElasticsearchConfig captures connection details for the upstream Elasticsearch cluster.
type ElasticsearchConfig struct {
	Addresses        []string
	APIKey           string
	Username         string
	Password         string
	CloudID          string
	Index            string
	EmbeddingDim     int
	RequestTimeout   time.Duration
	SkipIndexBootstr bool
	AllowMissing     bool
}

// elasticsearchClient wraps the official Go client and exposes the operations the
// search service needs. The type stays package-private so the rest of the codebase
// depends on the narrower interface.
type elasticsearchClient struct {
	client       *elasticsearch.Client
	index        string
	embeddingDim int
	timeout      time.Duration
}

// elasticsearchHit represents the minimal information the caller requires from a search response.
type elasticsearchHit struct {
	ID     string
	Score  float64
	Source map[string]interface{}
}

// newElasticsearchClient creates a client configured for the provided cluster and
// ensures the target index exists with the expected mapping.
func newElasticsearchClient(cfg ElasticsearchConfig) (*elasticsearchClient, error) {
	if len(cfg.Addresses) == 0 && cfg.CloudID == "" {
		cfg.Addresses = []string{"http://localhost:9200"}
	}
	if cfg.Index == "" {
		cfg.Index = "agenticaieth-docs"
	}
	if cfg.EmbeddingDim <= 0 {
		cfg.EmbeddingDim = 768
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 10 * time.Second
	}

	esCfg := elasticsearch.Config{
		Addresses: cfg.Addresses,
		APIKey:    cfg.APIKey,
		Username:  cfg.Username,
		Password:  cfg.Password,
		CloudID:   cfg.CloudID,
		Transport: &http.Transport{
			ResponseHeaderTimeout: cfg.RequestTimeout,
		},
	}

	client, err := elasticsearch.NewClient(esCfg)
	if err != nil {
		if cfg.AllowMissing {
			log.Printf("search: Elasticsearch client unavailable, falling back to in-memory index: %v", err)
			return nil, nil
		}
		return nil, fmt.Errorf("create elasticsearch client: %w", err)
	}

	ec := &elasticsearchClient{
		client:       client,
		index:        cfg.Index,
		embeddingDim: cfg.EmbeddingDim,
		timeout:      cfg.RequestTimeout,
	}

	if cfg.SkipIndexBootstr {
		return ec, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.RequestTimeout)
	defer cancel()
	if err := ec.ensureIndex(ctx); err != nil {
		if cfg.AllowMissing {
			log.Printf("search: Elasticsearch ensure index failed, falling back to in-memory index: %v", err)
			return nil, nil
		}
		return nil, err
	}

	return ec, nil
}

// ensureIndex creates the target index if it does not already exist.
func (c *elasticsearchClient) ensureIndex(ctx context.Context) error {
	exists, err := c.client.Indices.Exists([]string{c.index}, c.client.Indices.Exists.WithContext(ctx))
	if err != nil {
		return fmt.Errorf("check index existence: %w", err)
	}
	defer exists.Body.Close()

	if exists.StatusCode == http.StatusOK {
		return nil
	}
	if exists.StatusCode != http.StatusNotFound && exists.StatusCode != 0 {
		return fmt.Errorf("unexpected status checking index: %s", exists.String())
	}

	mapping := map[string]interface{}{
		"settings": map[string]interface{}{
			"number_of_shards":   1,
			"number_of_replicas": 0,
		},
		"mappings": map[string]interface{}{
			"properties": map[string]interface{}{
				"content": map[string]interface{}{"type": "text"},
				"title":   map[string]interface{}{"type": "text"},
				"task_type": map[string]interface{}{
					"type": "keyword",
				},
				"domain": map[string]interface{}{
					"type": "keyword",
				},
				"metadata": map[string]interface{}{
					"type": "object",
				},
				"embedding": map[string]interface{}{
					"type":       "dense_vector",
					"dims":       c.embeddingDim,
					"index":      true,
					"similarity": "cosine",
				},
				"created_at": map[string]interface{}{"type": "date"},
				"updated_at": map[string]interface{}{"type": "date"},
			},
		},
	}

	payload, err := json.Marshal(mapping)
	if err != nil {
		return fmt.Errorf("marshal index mapping: %w", err)
	}

	res, err := c.client.Indices.Create(
		c.index,
		c.client.Indices.Create.WithBody(bytes.NewReader(payload)),
		c.client.Indices.Create.WithContext(ctx),
	)
	if err != nil {
		return fmt.Errorf("create index: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("create index: %s", res.String())
	}

	return nil
}

// indexDocument upserts the supplied document into Elasticsearch.
func (c *elasticsearchClient) indexDocument(ctx context.Context, docID string, body map[string]interface{}) error {
	if docID == "" {
		return errors.New("document id is required")
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal document: %w", err)
	}

	res, err := c.client.Index(
		c.index,
		bytes.NewReader(payload),
		c.client.Index.WithDocumentID(docID),
		c.client.Index.WithContext(ctx),
		c.client.Index.WithRefresh("true"),
	)
	if err != nil {
		return fmt.Errorf("index document: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("index document: %s", res.String())
	}
	return nil
}

// deleteDocument removes the document from the index when it is deleted from HANA.
func (c *elasticsearchClient) deleteDocument(ctx context.Context, docID string) error {
	if docID == "" {
		return errors.New("document id is required")
	}

	res, err := c.client.Delete(
		c.index,
		docID,
		c.client.Delete.WithContext(ctx),
		c.client.Delete.WithRefresh("true"),
	)
	if err != nil {
		return fmt.Errorf("delete document: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode == http.StatusNotFound {
		return nil
	}

	if res.IsError() {
		return fmt.Errorf("delete document: %s", res.String())
	}
	return nil
}

// searchSimilarDocuments runs a vector KNN search and returns the top hits.
func (c *elasticsearchClient) searchSimilarDocuments(ctx context.Context, vector []float64, topK int, filters map[string]string) ([]elasticsearchHit, error) {
	if topK <= 0 {
		topK = 10
	}
	numCandidates := int(math.Max(float64(topK*4), 50))

	body := map[string]interface{}{
		"size": topK,
		"knn": map[string]interface{}{
			"field":          "embedding",
			"query_vector":   vector,
			"k":              topK,
			"num_candidates": numCandidates,
		},
	}

	if len(filters) > 0 {
		filterClauses := make([]map[string]interface{}, 0, len(filters))
		for key, value := range filters {
			if strings.TrimSpace(key) == "" || strings.TrimSpace(value) == "" {
				continue
			}
			filterClauses = append(filterClauses, map[string]interface{}{
				"term": map[string]interface{}{key: value},
			})
		}
		if len(filterClauses) > 0 {
			body["query"] = map[string]interface{}{
				"bool": map[string]interface{}{
					"filter": filterClauses,
				},
			}
		}
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal search body: %w", err)
	}

	res, err := c.client.Search(
		c.client.Search.WithContext(ctx),
		c.client.Search.WithIndex(c.index),
		c.client.Search.WithBody(bytes.NewReader(payload)),
		c.client.Search.WithTrackTotalHits(false),
	)
	if err != nil {
		return nil, fmt.Errorf("execute search: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return nil, fmt.Errorf("search error: %s", res.String())
	}

	var parsed struct {
		Hits struct {
			Hits []struct {
				ID     string                 `json:"_id"`
				Score  float64                `json:"_score"`
				Source map[string]interface{} `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(res.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}

	hits := make([]elasticsearchHit, 0, len(parsed.Hits.Hits))
	for _, hit := range parsed.Hits.Hits {
		hits = append(hits, elasticsearchHit{
			ID:     hit.ID,
			Score:  hit.Score,
			Source: hit.Source,
		})
	}

	return hits, nil
}

// getDocument retrieves a document by ID from Elasticsearch.
func (c *elasticsearchClient) getDocument(ctx context.Context, docID string) (map[string]interface{}, error) {
	if docID == "" {
		return nil, errors.New("document id is required")
	}

	res, err := c.client.Get(
		c.index,
		docID,
		c.client.Get.WithContext(ctx),
	)
	if err != nil {
		return nil, fmt.Errorf("get document: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if res.IsError() {
		return nil, fmt.Errorf("get document: %s", res.String())
	}

	var parsed struct {
		Source map[string]interface{} `json:"_source"`
	}

	if err := json.NewDecoder(res.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode get response: %w", err)
	}

	return parsed.Source, nil
}

// searchByFilters runs a filter-only query (no vector) for fetching documents by metadata.
func (c *elasticsearchClient) searchByFilters(ctx context.Context, filters map[string]string, size int) ([]elasticsearchHit, error) {
	if size <= 0 {
		size = 10
	}

	filterClauses := make([]map[string]interface{}, 0, len(filters))
	for key, value := range filters {
		if strings.TrimSpace(key) == "" || strings.TrimSpace(value) == "" {
			continue
		}
		filterClauses = append(filterClauses, map[string]interface{}{
			"term": map[string]interface{}{key: value},
		})
	}

	body := map[string]interface{}{
		"size": size,
		"query": map[string]interface{}{
			"bool": map[string]interface{}{
				"filter": filterClauses,
			},
		},
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal filter search: %w", err)
	}

	res, err := c.client.Search(
		c.client.Search.WithContext(ctx),
		c.client.Search.WithIndex(c.index),
		c.client.Search.WithBody(bytes.NewReader(payload)),
		c.client.Search.WithTrackTotalHits(false),
	)
	if err != nil {
		return nil, fmt.Errorf("execute filter search: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return nil, fmt.Errorf("filter search error: %s", res.String())
	}

	var parsed struct {
		Hits struct {
			Hits []struct {
				ID     string                 `json:"_id"`
				Score  float64                `json:"_score"`
				Source map[string]interface{} `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(res.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode filter search: %w", err)
	}

	results := make([]elasticsearchHit, 0, len(parsed.Hits.Hits))
	for _, hit := range parsed.Hits.Hits {
		results = append(results, elasticsearchHit{
			ID:     hit.ID,
			Score:  hit.Score,
			Source: hit.Source,
		})
	}

	return results, nil
}

// countDocuments returns the total documents stored in the index.
func (c *elasticsearchClient) countDocuments(ctx context.Context) (int64, error) {
	res, err := c.client.Count(
		c.client.Count.WithContext(ctx),
		c.client.Count.WithIndex(c.index),
	)
	if err != nil {
		return 0, fmt.Errorf("count documents: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return 0, fmt.Errorf("count documents: %s", res.String())
	}

	var parsed struct {
		Count int64 `json:"count"`
	}

	if err := json.NewDecoder(res.Body).Decode(&parsed); err != nil {
		return 0, fmt.Errorf("decode count response: %w", err)
	}

	return parsed.Count, nil
}
