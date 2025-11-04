package testing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// HTTPExtractClient implements ExtractClient using HTTP API.
type HTTPExtractClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewHTTPExtractClient creates a new HTTP extract client.
func NewHTTPExtractClient(baseURL string) *HTTPExtractClient {
	return &HTTPExtractClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// QueryKnowledgeGraph executes a Cypher query against the knowledge graph.
func (c *HTTPExtractClient) QueryKnowledgeGraph(query string, params map[string]any) ([]map[string]any, error) {
	url := fmt.Sprintf("%s/knowledge-graph/query", c.baseURL)
	
	payload := map[string]any{
		"query": query,
	}
	if params != nil {
		payload["params"] = params
	}
	
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewReader(body))
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
		return nil, fmt.Errorf("query failed with status %d", resp.StatusCode)
	}
	
	var result struct {
		Columns []string                   `json:"columns"`
		Data    []map[string]any           `json:"data"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	
	// Convert columns+data format to map format
	rows := make([]map[string]any, 0, len(result.Data))
	for _, rowData := range result.Data {
		row := make(map[string]any)
		for _, col := range result.Columns {
			if val, ok := rowData[col]; ok {
				row[col] = val
			}
		}
		rows = append(rows, row)
	}
	
	return rows, nil
}

// GetKnowledgeGraph gets the full knowledge graph.
func (c *HTTPExtractClient) GetKnowledgeGraph(projectID, systemID string) (*GraphData, error) {
	url := fmt.Sprintf("%s/knowledge-graph", c.baseURL)
	
	payload := map[string]any{
		"project_id": projectID,
	}
	if systemID != "" {
		payload["system_id"] = systemID
	}
	
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewReader(body))
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
		return nil, fmt.Errorf("get graph failed with status %d", resp.StatusCode)
	}
	
	var graphData GraphData
	if err := json.NewDecoder(resp.Body).Decode(&graphData); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	
	return &graphData, nil
}

