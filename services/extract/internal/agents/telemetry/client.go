package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"
)

// Client wraps access to the agent telemetry service hosted in the goose server.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient returns a telemetry client using the provided base URL. The URL must not be empty.
func NewClient(baseURL string, httpClient *http.Client) (*Client, error) {
	trimmed := strings.TrimSpace(baseURL)
	if trimmed == "" {
		return nil, fmt.Errorf("agent telemetry base URL is required")
	}

	// Ensure the base URL parses correctly up front so misconfiguration is immediately visible.
	if _, err := url.Parse(trimmed); err != nil {
		return nil, fmt.Errorf("invalid agent telemetry base URL: %w", err)
	}

	client := httpClient
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Second}
	}

	return &Client{
		baseURL:    strings.TrimRight(trimmed, "/"),
		httpClient: client,
	}, nil
}

// GetEvents fetches telemetry events for the provided session identifier.
func (c *Client) GetEvents(ctx context.Context, sessionID string) (*EventsResponse, error) {
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
		return nil, fmt.Errorf("telemetry service returned %d", resp.StatusCode)
	}

	var envelope EventsResponse
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return nil, fmt.Errorf("decode telemetry response: %w", err)
	}

	for i := range envelope.Events {
		if envelope.Events[i].Payload == nil {
			envelope.Events[i].Payload = make(map[string]any)
		}
	}

	return &envelope, nil
}

func (c *Client) buildURL(parts ...string) (string, error) {
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
