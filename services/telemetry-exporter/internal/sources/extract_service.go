package sources

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/telemetry-exporter/internal/exporter"
)

// ExtractServiceClient provides access to Extract service agent metrics.
type ExtractServiceClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewExtractServiceClient creates a new Extract service client.
func NewExtractServiceClient(baseURL string, timeout time.Duration) *ExtractServiceClient {
	return &ExtractServiceClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// GetSessionTelemetry fetches agent telemetry for a session from Extract service.
func (c *ExtractServiceClient) GetSessionTelemetry(ctx context.Context, sessionID string) (*exporter.ExtractServiceResponse, error) {
	url := fmt.Sprintf("%s/signavio/agent-metrics/%s", c.baseURL, sessionID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	var telemetry exporter.ExtractServiceResponse
	if err := json.NewDecoder(resp.Body).Decode(&telemetry); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &telemetry, nil
}

