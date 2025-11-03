package sdk

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"
)

// HTTPClient captures the subset of http.Client used by the SDK. Allows callers
// to inject custom transport, retry logic, etc.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// Config controls how the Go SDK communicates with a LangGraph server.
type Config struct {
	BaseURL    string
	APIKey     string
	HTTPClient HTTPClient
	Timeout    time.Duration
}

// Client is a lightweight facade over the LangGraph server API.
type Client struct {
	cfg        Config
	baseURL    *url.URL
	httpClient HTTPClient
}

// NewClient constructs a Client from the supplied configuration.
func NewClient(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.BaseURL) == "" {
		return nil, errors.New("sdk: BaseURL must be provided")
	}

	parsed, err := url.Parse(cfg.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("sdk: invalid BaseURL: %w", err)
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		client := &http.Client{}
		if cfg.Timeout > 0 {
			client.Timeout = cfg.Timeout
		}
		httpClient = client
	}

	return &Client{
		cfg:        cfg,
		baseURL:    parsed,
		httpClient: httpClient,
	}, nil
}

// Health performs a simple GET request to the server health endpoint.
func (c *Client) Health(ctx context.Context) error {
	req, err := c.newRequest(ctx, http.MethodGet, "/health", nil)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	return fmt.Errorf("sdk: health check failed with status %s", resp.Status)
}

// RunGraphRequest describes the payload sent to the server to execute a graph.
type RunGraphRequest struct {
	GraphID string      `json:"graph_id"`
	Input   interface{} `json:"input"`
}

// RunGraphResponse describes the data returned by the server after execution.
type RunGraphResponse struct {
	RunID  string      `json:"run_id"`
	Status string      `json:"status"`
	Output interface{} `json:"output,omitempty"`
}

// RunGraph executes the given graph on the remote LangGraph server.
func (c *Client) RunGraph(ctx context.Context, payload RunGraphRequest) (*RunGraphResponse, error) {
	if strings.TrimSpace(payload.GraphID) == "" {
		return nil, errors.New("sdk: graph id must be provided")
	}

	req, err := c.newRequest(ctx, http.MethodPost, "/api/graphs/run", payload)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("sdk: run graph failed with status %s", resp.Status)
	}

	var out RunGraphResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("sdk: decode response: %w", err)
	}
	return &out, nil
}

func (c *Client) newRequest(ctx context.Context, method, p string, body interface{}) (*http.Request, error) {
	u, err := c.resolveEndpoint(p)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			return nil, fmt.Errorf("sdk: encode request: %w", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, u.String(), &buf)
	if err != nil {
		return nil, err
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}
	return req, nil
}

func (c *Client) resolveEndpoint(p string) (*url.URL, error) {
	if p == "" {
		copyURL := *c.baseURL
		return &copyURL, nil
	}
	rel, err := url.Parse(p)
	if err != nil {
		return nil, fmt.Errorf("sdk: parse endpoint: %w", err)
	}
	if rel.IsAbs() {
		return rel, nil
	}
	joinedPath := rel.Path
	if basePath := strings.TrimSuffix(c.baseURL.Path, "/"); basePath != "" || joinedPath != "" {
		joinedPath = path.Join(basePath, strings.TrimPrefix(rel.Path, "/"))
	}
	if joinedPath == "" || !strings.HasPrefix(joinedPath, "/") {
		joinedPath = "/" + strings.TrimPrefix(joinedPath, "/")
	}
	resolved := *c.baseURL
	resolved.Path = joinedPath
	resolved.RawQuery = rel.RawQuery
	resolved.Fragment = rel.Fragment
	return &resolved, nil
}

// GetRunResponse contains status information for a specific run.
type GetRunResponse struct {
	RunID  string      `json:"run_id"`
	Status string      `json:"status"`
	Output interface{} `json:"output,omitempty"`
}

// GetRun fetches the status of a graph run.
func (c *Client) GetRun(ctx context.Context, runID string) (*GetRunResponse, error) {
	if strings.TrimSpace(runID) == "" {
		return nil, errors.New("sdk: run id must be provided")
	}

	endpoint := fmt.Sprintf("/api/runs/%s", runID)
	req, err := c.newRequest(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("sdk: get run failed with status %s", resp.Status)
	}

	var out GetRunResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("sdk: decode response: %w", err)
	}
	return &out, nil
}

// DeployRequest describes a deployment creation payload.
type DeployRequest struct {
	GraphID string `json:"graph_id"`
	Name    string `json:"name"`
}

// DeployResponse contains deployment metadata returned by the server.
type DeployResponse struct {
	DeploymentID string `json:"deployment_id"`
	Status       string `json:"status"`
}

// DeployGraph registers a graph as a deployment on the server.
func (c *Client) DeployGraph(ctx context.Context, payload DeployRequest) (*DeployResponse, error) {
	if strings.TrimSpace(payload.GraphID) == "" {
		return nil, errors.New("sdk: graph id must be provided")
	}
	if strings.TrimSpace(payload.Name) == "" {
		return nil, errors.New("sdk: deployment name must be provided")
	}

	req, err := c.newRequest(ctx, http.MethodPost, "/api/deployments", payload)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("sdk: deploy graph failed with status %s", resp.Status)
	}

	var out DeployResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("sdk: decode response: %w", err)
	}
	return &out, nil
}

// DeleteDeployment removes a deployment from the server.
func (c *Client) DeleteDeployment(ctx context.Context, deploymentID string) error {
	if strings.TrimSpace(deploymentID) == "" {
		return errors.New("sdk: deployment id must be provided")
	}

	endpoint := fmt.Sprintf("/api/deployments/%s", deploymentID)
	req, err := c.newRequest(ctx, http.MethodDelete, endpoint, nil)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("sdk: delete deployment failed with status %s", resp.Status)
	}
	return nil
}
