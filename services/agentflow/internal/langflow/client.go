package langflow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"
)

// Client provides a minimal HTTP client for interacting with a Langflow instance.
type Client struct {
	baseURL   *url.URL
	http      *http.Client
	headers   map[string]string
	userAgent string
}

// Option configures the Langflow client.
type Option func(*Client)

// WithHTTPClient overrides the default HTTP client.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(c *Client) {
		if httpClient != nil {
			c.http = httpClient
		}
	}
}

// WithAPIKey configures an X-API-Key header for requests.
func WithAPIKey(key string) Option {
	return func(c *Client) {
		if strings.TrimSpace(key) == "" {
			return
		}
		c.headers["X-API-Key"] = key
	}
}

// WithAuthToken configures an Authorization bearer token header.
func WithAuthToken(token string) Option {
	return func(c *Client) {
		if strings.TrimSpace(token) == "" {
			return
		}
		c.headers["Authorization"] = "Bearer " + token
	}
}

// WithStaticHeader adds a static header key/value pair for all requests.
func WithStaticHeader(key, value string) Option {
	return func(c *Client) {
		if strings.TrimSpace(key) == "" {
			return
		}
		if c.headers == nil {
			c.headers = map[string]string{}
		}
		c.headers[key] = value
	}
}

// WithUserAgent configures a custom User-Agent header.
func WithUserAgent(agent string) Option {
	return func(c *Client) {
		c.userAgent = strings.TrimSpace(agent)
	}
}

// NewClient constructs a new Langflow client.
func NewClient(rawURL string, opts ...Option) (*Client, error) {
	if strings.TrimSpace(rawURL) == "" {
		return nil, errors.New("base url is required")
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parse base url: %w", err)
	}
	parsed.Path = strings.TrimSuffix(parsed.Path, "/")

	client := &Client{
		baseURL: parsed,
		http: &http.Client{
			Timeout: 60 * time.Second,
		},
		headers: map[string]string{},
	}

	for _, opt := range opts {
		opt(client)
	}

	if client.userAgent == "" {
		client.userAgent = "agenticAiETH-agentflow/0.1"
	}

	return client, nil
}

// ListFlows returns metadata for all flows accessible to the caller.
func (c *Client) ListFlows(ctx context.Context) ([]FlowRecord, error) {
	var payload any
	if err := c.do(ctx, http.MethodGet, "/api/v1/flows/", nil, &payload); err != nil {
		return nil, err
	}
	return decodeFlowList(payload)
}

// Version returns the semantic version string exposed by the Langflow server.
func (c *Client) Version(ctx context.Context) (string, error) {
	var payload map[string]any
	if err := c.do(ctx, http.MethodGet, "/api/v1/version", nil, &payload); err != nil {
		return "", err
	}
	if version, ok := payload["version"].(string); ok {
		return version, nil
	}
	return "", fmt.Errorf("unexpected version payload: %v", payload)
}

// GetFlow returns a single flow definition by identifier.
func (c *Client) GetFlow(ctx context.Context, flowID string) (FlowRecord, error) {
	if strings.TrimSpace(flowID) == "" {
		return FlowRecord{}, errors.New("flow id is required")
	}
	var payload any
	path := fmt.Sprintf("/api/v1/flows/%s", url.PathEscape(flowID))
	if err := c.do(ctx, http.MethodGet, path, nil, &payload); err != nil {
		return FlowRecord{}, err
	}
	rec, err := decodeFlow(payload)
	if err != nil {
		return FlowRecord{}, err
	}
	return rec, nil
}

// ImportFlow uploads or overwrites a flow definition.
func (c *Client) ImportFlow(ctx context.Context, req FlowImportRequest) (FlowRecord, error) {
	if len(req.Flow) == 0 {
		return FlowRecord{}, errors.New("flow import request missing flow payload")
	}
	if req.Force && strings.TrimSpace(req.RemoteID) != "" {
		deletePath := fmt.Sprintf("/api/v1/flows/%s", url.PathEscape(req.RemoteID))
		if err := c.do(ctx, http.MethodDelete, deletePath, nil, nil); err != nil {
			if apiErr, ok := err.(*APIError); !ok || apiErr.StatusCode != http.StatusNotFound {
				return FlowRecord{}, fmt.Errorf("delete existing flow: %w", err)
			}
		}
	}
	var flowPayload map[string]any
	if err := json.Unmarshal(req.Flow, &flowPayload); err != nil {
		return FlowRecord{}, fmt.Errorf("decode flow payload: %w", err)
	}
	if req.Force {
		if name, ok := flowPayload["name"].(string); ok && strings.TrimSpace(name) != "" {
			if flows, err := c.ListFlows(ctx); err == nil {
				for _, candidate := range flows {
					if candidate.Name == name {
						deletePath := fmt.Sprintf("/api/v1/flows/%s", url.PathEscape(candidate.ID))
						if err := c.do(ctx, http.MethodDelete, deletePath, nil, nil); err != nil {
							if apiErr, ok := err.(*APIError); !ok || apiErr.StatusCode != http.StatusNotFound {
								return FlowRecord{}, fmt.Errorf("delete existing flow: %w", err)
							}
						}
					}
				}
			}
		}
	}
	if req.ProjectID != "" {
		if _, exists := flowPayload["project_id"]; !exists {
			flowPayload["project_id"] = req.ProjectID
		}
	}
	if req.FolderPath != "" {
		if _, exists := flowPayload["folder_path"]; !exists {
			flowPayload["folder_path"] = req.FolderPath
		}
	}
	if req.Force {
		flowPayload["force"] = true
	}

	requestBody := map[string]any{
		"flows": []any{flowPayload},
	}

	var payload any
	if err := c.do(ctx, http.MethodPost, "/api/v1/flows/batch/", requestBody, &payload); err != nil {
		return FlowRecord{}, err
	}
	records, err := decodeFlowList(payload)
	if err != nil {
		return FlowRecord{}, err
	}
	if len(records) == 0 {
		return FlowRecord{}, errors.New("langflow import returned no records")
	}
	return records[0], nil
}

// RunFlow executes a flow by identifier with the provided payload.
func (c *Client) RunFlow(ctx context.Context, flowID string, request RunFlowRequest) (RunFlowResult, error) {
	if strings.TrimSpace(flowID) == "" {
		return RunFlowResult{}, errors.New("flow id is required")
	}
	var payload map[string]any
	path := fmt.Sprintf("/api/v1/run/%s", url.PathEscape(flowID))
	if err := c.do(ctx, http.MethodPost, path, request, &payload); err != nil {
		return RunFlowResult{}, err
	}
	return RunFlowResult{Raw: payload}, nil
}

func (c *Client) do(ctx context.Context, method, relPath string, body any, out any) error {
	if ctx == nil {
		ctx = context.Background()
	}

	u := *c.baseURL
	rel := strings.TrimPrefix(relPath, "/")
	u.Path = path.Join(c.baseURL.Path, rel)

	var bodyReader io.Reader
	if body != nil {
		buf := &bytes.Buffer{}
		encoder := json.NewEncoder(buf)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(body); err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		bodyReader = buf
	}

	req, err := http.NewRequestWithContext(ctx, method, u.String(), bodyReader)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if bodyReader != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.userAgent != "" {
		req.Header.Set("User-Agent", c.userAgent)
	}
	for key, value := range c.headers {
		req.Header.Set(key, value)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("perform request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		apiErr := &APIError{StatusCode: resp.StatusCode, Body: respBody}
		var parsed map[string]any
		if json.Unmarshal(respBody, &parsed) == nil {
			if msg, ok := parsed["detail"].(string); ok && msg != "" {
				apiErr.Message = msg
			} else if msg, ok := parsed["message"].(string); ok && msg != "" {
				apiErr.Message = msg
			}
		}
		return apiErr
	}

	if out == nil {
		return nil
	}

	if len(respBody) == 0 {
		return nil
	}

	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	return nil
}

func decodeFlowList(payload any) ([]FlowRecord, error) {
	switch v := payload.(type) {
	case nil:
		return nil, nil
	case []any:
		return decodeFlowArray(v)
	case map[string]any:
		if flows, ok := v["flows"]; ok {
			return decodeFlowList(flows)
		}
		if data, ok := v["data"]; ok {
			return decodeFlowList(data)
		}
		if items, ok := v["items"]; ok {
			return decodeFlowList(items)
		}
		return decodeFlowArray([]any{v})
	default:
		return nil, fmt.Errorf("unexpected flows payload of type %T", payload)
	}
}

func decodeFlowArray(items []any) ([]FlowRecord, error) {
	results := make([]FlowRecord, 0, len(items))
	for _, item := range items {
		rec, err := decodeFlow(item)
		if err != nil {
			return nil, err
		}
		results = append(results, rec)
	}
	return results, nil
}

func decodeFlow(payload any) (FlowRecord, error) {
	data, ok := payload.(map[string]any)
	if !ok || data == nil {
		return FlowRecord{}, fmt.Errorf("invalid flow payload: expected object, got %T", payload)
	}

	record := FlowRecord{
		Raw: data,
	}

	if id, ok := data["id"].(string); ok {
		record.ID = id
	}
	if name, ok := data["name"].(string); ok {
		record.Name = name
	}
	if desc, ok := data["description"].(string); ok {
		record.Description = desc
	}
	if projectID, ok := data["project_id"].(string); ok {
		record.ProjectID = projectID
	}
	switch updated := data["updated_at"].(type) {
	case string:
		if ts, err := time.Parse(time.RFC3339, updated); err == nil {
			record.UpdatedAt = ts
		}
	}

	return record, nil
}
