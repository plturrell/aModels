package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// DeepAgentsClient handles communication with the DeepAgents service.
type DeepAgentsClient struct {
	baseURL string
	client  *http.Client
	logger  *log.Logger
	enabled bool
}

// NewDeepAgentsClient creates a new DeepAgents client.
// DeepAgents is enabled by default. Set DEEPAGENTS_ENABLED=false to disable.
func NewDeepAgentsClient(logger *log.Logger) *DeepAgentsClient {
	baseURL := os.Getenv("DEEPAGENTS_URL")
	if baseURL == "" {
		baseURL = "http://deepagents-service:9004"
	}

	// Enabled by default (10/10 integration)
	// Only disable if explicitly set to false
	enabled := os.Getenv("DEEPAGENTS_ENABLED") != "false"

	if enabled {
		logger.Printf("DeepAgents integration enabled (URL: %s)", baseURL)
	} else {
		logger.Printf("DeepAgents integration disabled (DEEPAGENTS_ENABLED=false)")
	}

	return &DeepAgentsClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 120 * time.Second, // Deep agents can take longer
		},
		logger:  logger,
		enabled: enabled,
	}
}

// AnalyzeGraphRequest represents a request to analyze a knowledge graph.
type AnalyzeGraphRequest struct {
	Messages []Message `json:"messages"`
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AnalyzeGraphResponse represents the response from DeepAgents.
type AnalyzeGraphResponse struct {
	Messages         []Message              `json:"messages"`
	StructuredOutput map[string]interface{}  `json:"structured_output,omitempty"`
	Result           any                     `json:"result,omitempty"`
}

// GraphAnalysisResult represents structured analysis results.
type GraphAnalysisResult struct {
	KeyFindings      []string `json:"key_findings"`
	QualityObservations []string `json:"quality_observations"`
	Recommendations []string `json:"recommendations"`
	Issues          []string `json:"issues"`
	Optimizations   []string `json:"optimizations"`
	SchemaQuality   float64  `json:"schema_quality,omitempty"`
	LineageInsights []string `json:"lineage_insights,omitempty"`
}

// AnalyzeKnowledgeGraph analyzes a knowledge graph using DeepAgents.
// This provides AI-powered insights, recommendations, and analysis of the graph structure.
// Returns nil, nil if disabled or if service is unavailable (non-fatal).
//
// Error Handling Pattern:
// - Non-fatal integration: Returns nil, nil on failure (graceful degradation)
// - Retry logic: 2 retries with exponential backoff (1s, 2s)
// - Health check: 5s timeout before attempting analysis
// - Request timeout: 120s for agent operations
// - Retries on: network errors, 5xx server errors
// - Does not retry: 4xx client errors (returns nil, nil)
func (c *DeepAgentsClient) AnalyzeKnowledgeGraph(ctx context.Context, graphSummary string, projectID, systemID string) (*AnalyzeGraphResponse, error) {
	if !c.enabled {
		return nil, nil
	}

	// Quick health check before attempting analysis
	healthCtx, healthCancel := context.WithTimeout(ctx, 5*time.Second)
	defer healthCancel()
	if !c.checkHealth(healthCtx) {
		c.logger.Printf("DeepAgents service unavailable, skipping analysis")
		return nil, nil
	}

	// Build analysis prompt
	prompt := fmt.Sprintf(`Analyze this knowledge graph and provide insights:

%s

Context:
- Project ID: %s
- System ID: %s

Please provide structured analysis with:
1. Key findings about the graph structure
2. Data quality observations
3. Recommendations for improvements
4. Potential issues or inconsistencies
5. Suggestions for optimization
6. Schema quality assessment
7. Data lineage insights

Be specific and actionable.`, graphSummary, projectID, systemID)

	// Define JSON schema for structured output
	jsonSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"key_findings": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"quality_observations": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"recommendations": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"issues": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"optimizations": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"schema_quality": map[string]interface{}{
				"type":    "number",
				"minimum": 0,
				"maximum": 1,
			},
			"lineage_insights": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
		},
		"required": []string{"key_findings", "quality_observations", "recommendations", "issues", "optimizations"},
	}

	request := map[string]interface{}{
		"messages": []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		"response_format": map[string]interface{}{
			"type":       "json_schema",
			"json_schema": jsonSchema,
		},
	}

	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/invoke/structured", c.baseURL)

	// Retry logic for resilience
	var lastErr error
	maxRetries := 2
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			backoff := time.Duration(attempt) * time.Second
			c.logger.Printf("Retrying DeepAgents request (attempt %d/%d) after %v", attempt+1, maxRetries+1, backoff)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("request failed: %w", err)
			if attempt < maxRetries {
				continue // Retry on network errors
			}
			c.logger.Printf("DeepAgents request failed after %d attempts: %v", attempt+1, lastErr)
			return nil, nil // Return nil, nil (non-fatal) instead of error
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes := make([]byte, 4096)
			n, _ := resp.Body.Read(bodyBytes)
			lastErr = fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(bodyBytes[:n]))
			if resp.StatusCode >= 500 && attempt < maxRetries {
				continue // Retry on server errors
			}
			c.logger.Printf("DeepAgents returned status %d: %s", resp.StatusCode, string(bodyBytes[:n]))
			return nil, nil // Return nil, nil (non-fatal) instead of error
		}

		var response struct {
			Messages         []Message             `json:"messages"`
			StructuredOutput map[string]interface{} `json:"structured_output"`
			Result           interface{}           `json:"result,omitempty"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
			lastErr = fmt.Errorf("decode response: %w", err)
			if attempt < maxRetries {
				continue // Retry on decode errors
			}
			c.logger.Printf("DeepAgents decode failed after %d attempts: %v", attempt+1, lastErr)
			return nil, nil // Return nil, nil (non-fatal) instead of error
		}

		// Convert to AnalyzeGraphResponse
		result := &AnalyzeGraphResponse{
			Messages:         response.Messages,
			StructuredOutput: response.StructuredOutput,
			Result:           response.Result,
		}

		c.logger.Printf("DeepAgents analysis completed successfully (attempt %d)", attempt+1)
		return result, nil
	}

	// Should not reach here, but handle gracefully
	c.logger.Printf("DeepAgents analysis failed after all retries: %v", lastErr)
	return nil, nil
}

// AnalyzeSchemaQuality analyzes schema quality using DeepAgents.
func (c *DeepAgentsClient) AnalyzeSchemaQuality(ctx context.Context, schemaInfo string, projectID, systemID string) (map[string]interface{}, error) {
	if !c.enabled {
		return nil, nil
	}

	healthCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if !c.checkHealth(healthCtx) {
		return nil, nil
	}

	prompt := fmt.Sprintf(`Analyze schema quality for this system:

%s

Context:
- Project ID: %s
- System ID: %s

Provide structured analysis with:
1. Schema completeness score (0-1)
2. Naming consistency assessment
3. Data type appropriateness
4. Missing constraints or relationships
5. Recommendations for improvement`, schemaInfo, projectID, systemID)

	jsonSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"completeness_score": map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
			"naming_consistency": map[string]interface{}{"type": "string"},
			"data_type_issues": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"missing_constraints": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"recommendations": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
		},
	}

	return c.callStructuredEndpoint(ctx, prompt, jsonSchema)
}

// AnalyzeDataLineage analyzes data lineage using DeepAgents.
func (c *DeepAgentsClient) AnalyzeDataLineage(ctx context.Context, lineageInfo string, projectID, systemID string) (map[string]interface{}, error) {
	if !c.enabled {
		return nil, nil
	}

	healthCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if !c.checkHealth(healthCtx) {
		return nil, nil
	}

	prompt := fmt.Sprintf(`Analyze data lineage for this system:

%s

Context:
- Project ID: %s
- System ID: %s

Provide structured analysis with:
1. Lineage completeness
2. Data flow patterns
3. Potential bottlenecks
4. Missing lineage connections
5. Recommendations`, lineageInfo, projectID, systemID)

	jsonSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"completeness": map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
			"flow_patterns": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"bottlenecks": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"missing_connections": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"recommendations": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{"type": "string"},
			},
		},
	}

	return c.callStructuredEndpoint(ctx, prompt, jsonSchema)
}

// SuggestCrossSystemMappings suggests cross-system mappings using DeepAgents.
func (c *DeepAgentsClient) SuggestCrossSystemMappings(ctx context.Context, sourceSchema, targetSchema string, projectID string) (map[string]interface{}, error) {
	if !c.enabled {
		return nil, nil
	}

	healthCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if !c.checkHealth(healthCtx) {
		return nil, nil
	}

	prompt := fmt.Sprintf(`Suggest cross-system mappings between source and target schemas:

Source Schema:
%s

Target Schema:
%s

Project ID: %s

Provide structured mapping suggestions with:
1. Field-to-field mappings
2. Transformation requirements
3. Data type conversions needed
4. Confidence scores for each mapping
5. Potential issues or warnings`, sourceSchema, targetSchema, projectID)

	jsonSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"mappings": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"source_field": map[string]interface{}{"type": "string"},
						"target_field": map[string]interface{}{"type": "string"},
						"confidence":   map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
						"transformation": map[string]interface{}{"type": "string"},
						"warnings": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{"type": "string"},
						},
					},
				},
			},
			"overall_confidence": map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
		},
	}

	return c.callStructuredEndpoint(ctx, prompt, jsonSchema)
}

// callStructuredEndpoint is a helper to call the structured endpoint.
func (c *DeepAgentsClient) callStructuredEndpoint(ctx context.Context, prompt string, jsonSchema map[string]interface{}) (map[string]interface{}, error) {
	request := map[string]interface{}{
		"messages": []Message{
			{Role: "user", Content: prompt},
		},
		"response_format": map[string]interface{}{
			"type":       "json_schema",
			"json_schema": jsonSchema,
		},
	}

	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/invoke/structured", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		c.logger.Printf("DeepAgents structured request failed: %v", err)
		return nil, nil // Non-fatal
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.logger.Printf("DeepAgents returned status %d", resp.StatusCode)
		return nil, nil // Non-fatal
	}

	var response struct {
		StructuredOutput map[string]interface{} `json:"structured_output"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		c.logger.Printf("Failed to decode DeepAgents response: %v", err)
		return nil, nil // Non-fatal
	}

	return response.StructuredOutput, nil
}

// checkHealth performs a quick health check on the DeepAgents service.
func (c *DeepAgentsClient) checkHealth(ctx context.Context) bool {
	endpoint := fmt.Sprintf("%s/healthz", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return false
	}

	healthClient := &http.Client{Timeout: 5 * time.Second}
	resp, err := healthClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

// FormatGraphSummary formats a knowledge graph for analysis.
func FormatGraphSummary(nodes []Node, edges []Edge, quality map[string]any, metrics map[string]any) string {
	var summary strings.Builder

	summary.WriteString(fmt.Sprintf("Knowledge Graph Summary:\n"))
	summary.WriteString(fmt.Sprintf("- Nodes: %d\n", len(nodes)))
	summary.WriteString(fmt.Sprintf("- Edges: %d\n", len(edges)))

	// Count node types
	nodeTypes := make(map[string]int)
	for _, node := range nodes {
		nodeTypes[node.Type]++
	}
	summary.WriteString("\nNode Types:\n")
	for nodeType, count := range nodeTypes {
		summary.WriteString(fmt.Sprintf("  - %s: %d\n", nodeType, count))
	}

	// Count edge labels
	edgeLabels := make(map[string]int)
	for _, edge := range edges {
		edgeLabels[edge.Label]++
	}
	summary.WriteString("\nEdge Labels:\n")
	for label, count := range edgeLabels {
		summary.WriteString(fmt.Sprintf("  - %s: %d\n", label, count))
	}

	// Add quality metrics
	if quality != nil {
		if score, ok := quality["score"].(float64); ok {
			summary.WriteString(fmt.Sprintf("\nQuality Score: %.2f\n", score))
		}
		if level, ok := quality["level"].(string); ok {
			summary.WriteString(fmt.Sprintf("Quality Level: %s\n", level))
		}
		if issues, ok := quality["issues"].([]any); ok && len(issues) > 0 {
			summary.WriteString("\nIssues:\n")
			for _, issue := range issues {
				summary.WriteString(fmt.Sprintf("  - %v\n", issue))
			}
		}
	}

	// Add information theory metrics
	if metrics != nil {
		summary.WriteString("\nInformation Theory Metrics:\n")
		if entropy, ok := metrics["metadata_entropy"].(float64); ok {
			summary.WriteString(fmt.Sprintf("  - Metadata Entropy: %.2f\n", entropy))
		}
		if klDiv, ok := metrics["kl_divergence"].(float64); ok {
			summary.WriteString(fmt.Sprintf("  - KL Divergence: %.2f\n", klDiv))
		}
		if colCount, ok := metrics["column_count"].(float64); ok {
			summary.WriteString(fmt.Sprintf("  - Column Count: %.0f\n", colCount))
		}
	}

	return summary.String()
}

