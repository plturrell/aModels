package sources

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"
)

// AgentTelemetryClient provides access to agent telemetry service.
type AgentTelemetryClient struct {
	baseURL    string
	httpClient *http.Client
}

// EventsResponse represents the response from agent telemetry service.
type EventsResponse struct {
	SessionID string             `json:"session_id"`
	Events    []AgentMetricEvent `json:"events"`
}

// AgentMetricEvent represents a telemetry event.
type AgentMetricEvent struct {
	Timestamp time.Time            `json:"timestamp"`
	SessionID string               `json:"session_id"`
	Type      string               `json:"type"`
	Payload   map[string]any       `json:"payload,omitempty"`
}

// NewAgentTelemetryClient creates a new agent telemetry client.
func NewAgentTelemetryClient(baseURL string, timeout time.Duration) (*AgentTelemetryClient, error) {
	trimmed := strings.TrimSpace(baseURL)
	if trimmed == "" {
		return nil, fmt.Errorf("agent telemetry base URL is required")
	}

	// Ensure the base URL parses correctly
	if _, err := url.Parse(trimmed); err != nil {
		return nil, fmt.Errorf("invalid agent telemetry base URL: %w", err)
	}

	return &AgentTelemetryClient{
		baseURL: strings.TrimRight(trimmed, "/"),
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// GetEvents fetches telemetry events for a session.
func (c *AgentTelemetryClient) GetEvents(ctx context.Context, sessionID string) (*EventsResponse, error) {
	if strings.TrimSpace(sessionID) == "" {
		return nil, fmt.Errorf("session ID is required")
	}

	endpoint, err := c.buildURL("/agent_metrics/", sessionID, "events")
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("create telemetry request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call telemetry service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("telemetry service returned %d: %s", resp.StatusCode, string(body))
	}

	var envelope EventsResponse
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return nil, fmt.Errorf("decode telemetry response: %w", err)
	}

	// Ensure payloads are initialized
	for i := range envelope.Events {
		if envelope.Events[i].Payload == nil {
			envelope.Events[i].Payload = make(map[string]any)
		}
	}

	return &envelope, nil
}

func (c *AgentTelemetryClient) buildURL(parts ...string) (string, error) {
	u, err := url.Parse(c.baseURL)
	if err != nil {
		return "", fmt.Errorf("parse base url: %w", err)
	}
	combined := path.Join(parts...)
	// path.Join strips trailing slash; ensure we maintain consistent path.
	if !strings.HasPrefix(combined, "/") {
		combined = "/" + combined
	}
	u.Path = strings.TrimSuffix(u.Path, "/") + combined
	return u.String(), nil
}

