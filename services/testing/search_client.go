package testing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// SearchClient provides search capabilities for test scenarios and patterns.
type SearchClient struct {
	baseURL    string
	httpClient *http.Client
	logger     *log.Logger
	enabled    bool
}

// NewSearchClient creates a new search client.
func NewSearchClient(baseURL string, timeout time.Duration, enabled bool, logger *log.Logger) *SearchClient {
	if !enabled || baseURL == "" {
		return &SearchClient{
			enabled: false,
			logger:  logger,
		}
	}

	return &SearchClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		logger:  logger,
		enabled: true,
	}
}

// IsEnabled returns whether search is enabled.
func (c *SearchClient) IsEnabled() bool {
	return c.enabled
}

// SearchResult represents a search result.
type SearchResult struct {
	ID          string                 `json:"id"`
	Content     string                 `json:"content"`
	Similarity  float64                `json:"similarity"`
	Metadata    map[string]any         `json:"metadata"`
	ArtifactType string                `json:"artifact_type,omitempty"`
}

// SearchScenarios searches for similar test scenarios.
func (c *SearchClient) SearchScenarios(ctx context.Context, query string, limit int) ([]*TestScenario, error) {
	if !c.IsEnabled() {
		return nil, fmt.Errorf("search is not enabled")
	}

	// Use knowledge graph search to find related tables/processes
	results, err := c.SearchKnowledgeGraph(ctx, query, "table", limit)
	if err != nil {
		return nil, fmt.Errorf("search knowledge graph: %w", err)
	}

	// Convert search results to test scenarios
	scenarios := make([]*TestScenario, 0, len(results))
	for _, result := range results {
		// Extract table name from metadata
		tableName, ok := result.Metadata["label"].(string)
		if !ok {
			tableName, _ = result.Metadata["name"].(string)
		}

		if tableName != "" {
			scenario := &TestScenario{
				ID:          fmt.Sprintf("scenario_%s_%d", tableName, time.Now().Unix()),
				Name:        fmt.Sprintf("Test Scenario: %s", tableName),
				Description: fmt.Sprintf("Auto-generated from search: %s", query),
				Tables: []*TableTestConfig{
					{
						TableName: tableName,
						RowCount:  100, // Default
					},
				},
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
			}
			scenarios = append(scenarios, scenario)
		}
	}

	return scenarios, nil
}

// SearchPatterns searches for similar data patterns.
func (c *SearchClient) SearchPatterns(ctx context.Context, tableName, columnName string, limit int) ([]*ColumnPattern, error) {
	if !c.IsEnabled() {
		return nil, fmt.Errorf("search is not enabled")
	}

	query := fmt.Sprintf("column %s in table %s", columnName, tableName)
	results, err := c.SearchKnowledgeGraph(ctx, query, "column", limit)
	if err != nil {
		return nil, fmt.Errorf("search knowledge graph: %w", err)
	}

	patterns := make([]*ColumnPattern, 0, len(results))
	for _, result := range results {
		// Extract column info from metadata
		colName, _ := result.Metadata["label"].(string)
		if colName == "" {
			colName, _ = result.Metadata["name"].(string)
		}

		if colName != "" {
			pattern := &ColumnPattern{
				ColumnName:    colName,
				ValuePatterns: []string{},
				EnumValues:    []string{},
				CommonValues:  []any{},
			}

			// Extract patterns from metadata if available
			if patternsList, ok := result.Metadata["patterns"].([]any); ok {
				for _, p := range patternsList {
					if str, ok := p.(string); ok {
						pattern.ValuePatterns = append(pattern.ValuePatterns, str)
					}
				}
			}

			patterns = append(patterns, pattern)
		}
	}

	return patterns, nil
}

// SearchKnowledgeGraph performs semantic search on the knowledge graph.
func (c *SearchClient) SearchKnowledgeGraph(ctx context.Context, query string, artifactType string, limit int) ([]SearchResult, error) {
	if !c.IsEnabled() {
		return nil, fmt.Errorf("search is not enabled")
	}

	url := fmt.Sprintf("%s/knowledge-graph/search", c.baseURL)

	payload := map[string]any{
		"query":  query,
		"limit":  limit,
		"use_semantic": true,
	}

	if artifactType != "" {
		payload["artifact_type"] = artifactType
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search failed with status %d", resp.StatusCode)
	}

	var result struct {
		Results []SearchResult `json:"results"`
		Total   int            `json:"total,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return result.Results, nil
}

// SearchRelatedTables finds tables related to a given table.
func (c *SearchClient) SearchRelatedTables(ctx context.Context, tableName string, limit int) ([]string, error) {
	if !c.IsEnabled() {
		return nil, fmt.Errorf("search is not enabled")
	}

	query := fmt.Sprintf("tables related to %s", tableName)
	results, err := c.SearchKnowledgeGraph(ctx, query, "table", limit)
	if err != nil {
		return nil, fmt.Errorf("search related tables: %w", err)
	}

	tables := make([]string, 0, len(results))
	for _, result := range results {
		tableName, ok := result.Metadata["label"].(string)
		if !ok {
			tableName, _ = result.Metadata["name"].(string)
		}
		if tableName != "" {
			tables = append(tables, tableName)
		}
	}

	return tables, nil
}

