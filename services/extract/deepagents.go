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
	Messages []Message     `json:"messages"`
	Result   any           `json:"result,omitempty"`
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

Please provide:
1. Key findings about the graph structure
2. Data quality observations
3. Recommendations for improvements
4. Potential issues or inconsistencies
5. Suggestions for optimization

Be specific and actionable.`, graphSummary, projectID, systemID)

	request := AnalyzeGraphRequest{
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	}

	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/invoke", c.baseURL)

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

		var response AnalyzeGraphResponse
		if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
			lastErr = fmt.Errorf("decode response: %w", err)
			if attempt < maxRetries {
				continue // Retry on decode errors
			}
			c.logger.Printf("DeepAgents decode failed after %d attempts: %v", attempt+1, lastErr)
			return nil, nil // Return nil, nil (non-fatal) instead of error
		}

		c.logger.Printf("DeepAgents analysis completed successfully (attempt %d)", attempt+1)
		return &response, nil
	}

	// Should not reach here, but handle gracefully
	c.logger.Printf("DeepAgents analysis failed after all retries: %v", lastErr)
	return nil, nil
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

