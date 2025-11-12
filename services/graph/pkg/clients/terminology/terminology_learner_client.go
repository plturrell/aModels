package terminology

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

// HTTPTerminologyLearnerClient implements TerminologyLearnerInterface via HTTP calls to extract service.
type HTTPTerminologyLearnerClient struct {
	extractServiceURL string
	httpClient        *http.Client
	logger            *log.Logger
}

// NewHTTPTerminologyLearnerClient creates a new HTTP-based terminology learner client.
func NewHTTPTerminologyLearnerClient(extractServiceURL string, logger *log.Logger) *HTTPTerminologyLearnerClient {
	return &HTTPTerminologyLearnerClient{
		extractServiceURL: extractServiceURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		logger: logger,
	}
}

// convertTerminologyNodesToExtractFormat converts TerminologyNode to extract service format.
func (c *HTTPTerminologyLearnerClient) convertTerminologyNodesToExtractFormat(nodes []TerminologyNode) []map[string]interface{} {
	extractNodes := make([]map[string]interface{}, len(nodes))
	for i, node := range nodes {
		extractNodes[i] = map[string]interface{}{
			"id":    node.ID,
			"type":  node.Type,
			"label": node.Label,
			"props": node.Props,
		}
	}
	return extractNodes
}

// convertTerminologyEdgesToExtractFormat converts TerminologyEdge to extract service format.
func (c *HTTPTerminologyLearnerClient) convertTerminologyEdgesToExtractFormat(edges []TerminologyEdge) []map[string]interface{} {
	extractEdges := make([]map[string]interface{}, len(edges))
	for i, edge := range edges {
		extractEdges[i] = map[string]interface{}{
			"source": edge.SourceID,
			"target": edge.TargetID,
			"label":  edge.Label,
			"props":  edge.Props,
		}
	}
	return extractEdges
}

// LearnFromExtraction learns terminology from extraction nodes and edges via HTTP.
func (c *HTTPTerminologyLearnerClient) LearnFromExtraction(ctx context.Context, nodes []TerminologyNode, edges []TerminologyEdge) error {
	// Convert to format expected by extract service
	extractNodes := c.convertTerminologyNodesToExtractFormat(nodes)
	extractEdges := c.convertTerminologyEdgesToExtractFormat(edges)

	payload := map[string]interface{}{
		"nodes": extractNodes,
		"edges": extractEdges,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.extractServiceURL+"/terminology/learn", 
		strings.NewReader(string(data)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call extract service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("extract service returned status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// LearnDomain learns a domain term via HTTP.
func (c *HTTPTerminologyLearnerClient) LearnDomain(ctx context.Context, text, domain string, timestamp time.Time) error {
	payload := map[string]interface{}{
		"text":      text,
		"domain":    domain,
		"timestamp": timestamp.Format(time.RFC3339),
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.extractServiceURL+"/terminology/learn/domain",
		strings.NewReader(string(data)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call extract service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("extract service returned status %d", resp.StatusCode)
	}

	return nil
}

// LearnRole learns a role term via HTTP.
func (c *HTTPTerminologyLearnerClient) LearnRole(ctx context.Context, text, role string, timestamp time.Time) error {
	payload := map[string]interface{}{
		"text":      text,
		"role":      role,
		"timestamp": timestamp.Format(time.RFC3339),
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.extractServiceURL+"/terminology/learn/role",
		strings.NewReader(string(data)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call extract service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("extract service returned status %d", resp.StatusCode)
	}

	return nil
}

// InferDomain infers domain via HTTP.
func (c *HTTPTerminologyLearnerClient) InferDomain(ctx context.Context, columnName, tableName string, context map[string]interface{}) (string, float64) {
	payload := map[string]interface{}{
		"column_name": columnName,
		"table_name":  tableName,
		"context":     context,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to marshal infer domain request: %v", err)
		}
		return "unknown", 0.0
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.extractServiceURL+"/terminology/infer/domain",
		strings.NewReader(string(data)))
	if err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to create infer domain request: %v", err)
		}
		return "unknown", 0.0
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to call extract service for domain inference: %v", err)
		}
		return "unknown", 0.0
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if c.logger != nil {
			c.logger.Printf("Extract service returned status %d for domain inference: %s", resp.StatusCode, string(body))
		}
		return "unknown", 0.0
	}

	var result struct {
		Domain     string  `json:"domain"`
		Confidence float64 `json:"confidence"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to decode domain inference response: %v", err)
		}
		return "unknown", 0.0
	}

	return result.Domain, result.Confidence
}

// InferRole infers role via HTTP.
func (c *HTTPTerminologyLearnerClient) InferRole(ctx context.Context, columnName, columnType, tableName string, context map[string]interface{}) (string, float64) {
	payload := map[string]interface{}{
		"column_name": columnName,
		"column_type": columnType,
		"table_name":  tableName,
		"context":     context,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to marshal infer role request: %v", err)
		}
		return "unknown", 0.0
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.extractServiceURL+"/terminology/infer/role",
		strings.NewReader(string(data)))
	if err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to create infer role request: %v", err)
		}
		return "unknown", 0.0
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to call extract service for role inference: %v", err)
		}
		return "unknown", 0.0
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if c.logger != nil {
			c.logger.Printf("Extract service returned status %d for role inference: %s", resp.StatusCode, string(body))
		}
		return "unknown", 0.0
	}

	var result struct {
		Role       string  `json:"role"`
		Confidence float64 `json:"confidence"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to decode role inference response: %v", err)
		}
		return "unknown", 0.0
	}

	return result.Role, result.Confidence
}

// EnhanceEmbedding enhances embedding via HTTP.
func (c *HTTPTerminologyLearnerClient) EnhanceEmbedding(ctx context.Context, text string, baseEmbedding []float32, embeddingType string) ([]float32, error) {
	payload := map[string]interface{}{
		"text":          text,
		"base_embedding": baseEmbedding,
		"embedding_type": embeddingType,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.extractServiceURL+"/terminology/enhance/embedding",
		strings.NewReader(string(data)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call extract service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("extract service returned status %d", resp.StatusCode)
	}

	var result struct {
		EnhancedEmbedding []float32 `json:"enhanced_embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.EnhancedEmbedding, nil
}

