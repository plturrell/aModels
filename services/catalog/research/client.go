package research

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// DeepResearchClient is a Go client for the Open Deep Research service.
type DeepResearchClient struct {
	baseURL    string
	httpClient *http.Client
	logger     *log.Logger
}

// NewDeepResearchClient creates a new Deep Research client.
func NewDeepResearchClient(baseURL string, logger *log.Logger) *DeepResearchClient {
	if baseURL == "" {
		baseURL = "http://localhost:8085"
	}
	return &DeepResearchClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 300 * time.Second, // Research can take a long time
		},
		logger: logger,
	}
}

// ResearchRequest represents a research request.
type ResearchRequest struct {
	Query   string                 `json:"query"`
	Context map[string]interface{} `json:"context,omitempty"`
	Tools   []string               `json:"tools,omitempty"`
}

// ResearchReport represents a research report response.
type ResearchReport struct {
	Status   string                 `json:"status"`
	Report   *ReportContent         `json:"report,omitempty"`
	Error    string                 `json:"error,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ReportContent represents the content of a research report.
type ReportContent struct {
	Topic     string         `json:"topic"`
	Summary   string         `json:"summary"`
	Sections  []ReportSection `json:"sections"`
	Generated time.Time      `json:"generated"`
	Sources   []string       `json:"sources,omitempty"`
}

// ReportSection represents a section of a research report.
type ReportSection struct {
	Title   string   `json:"title"`
	Content string   `json:"content"`
	Sources []string `json:"sources,omitempty"`
}

// Research performs a research query using Open Deep Research.
func (c *DeepResearchClient) Research(ctx context.Context, req *ResearchRequest) (*ResearchReport, error) {
	// Prepare request payload
	payload := map[string]interface{}{
		"query": req.Query,
	}
	if req.Context != nil {
		payload["context"] = req.Context
	}
	if req.Tools != nil {
		payload["tools"] = req.Tools
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/research", c.baseURL), bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Log request
	if c.logger != nil {
		c.logger.Printf("Deep Research request: %s", req.Query)
	}

	// Execute request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		return &ResearchReport{
			Status: "error",
			Error:  fmt.Sprintf("service returned status %d: %s", resp.StatusCode, string(body)),
		}, nil
	}

	// Parse response
	var report ResearchReport
	if err := json.Unmarshal(body, &report); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Log success
	if c.logger != nil {
		c.logger.Printf("Deep Research completed: status=%s", report.Status)
	}

	return &report, nil
}

// ResearchMetadata performs metadata research for a given topic.
func (c *DeepResearchClient) ResearchMetadata(ctx context.Context, topic string, includeLineage, includeQuality bool) (*ResearchReport, error) {
	query := fmt.Sprintf(`
Research and document all metadata related to: %s

Include:
1. Data elements and their definitions
2. Data lineage (sources, transformations)
3. Quality metrics and SLOs
4. Access controls and permissions
5. Usage patterns and examples
6. Related data products
`, topic)

	req := &ResearchRequest{
		Query: query,
		Context: map[string]interface{}{
			"topic":          topic,
			"include_lineage": includeLineage,
			"include_quality": includeQuality,
		},
		Tools: []string{"sparql_query", "catalog_search"},
	}

	return c.Research(ctx, req)
}

// Health checks if the Deep Research service is healthy.
func (c *DeepResearchClient) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/healthz", c.baseURL), nil)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: status %d", resp.StatusCode)
	}

	return nil
}

