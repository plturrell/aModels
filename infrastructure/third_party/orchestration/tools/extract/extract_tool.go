package extract

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ServiceTool exposes the Layer4 extract service (entity extraction and training export)
// as a LangGraph tool.
//
// Required input format (JSON):
//
//	{
//	  "action": "extract",
//	  "document": "free-form text or file content",
//	  "prompt_description": "optional prompt override",
//	  "model_id": "optional model",
//	  "examples": [...]
//	}
//
// Training export example:
//
//	{
//	  "action": "training",
//	  "mode": "document",
//	  "document": {
//	    "inputs": ["/path/to/invoice.pdf"],
//	    "format": "markdown"
//	  }
//	}
//
// When action is omitted it defaults to "extract". The raw JSON response is returned.
type ServiceTool struct {
	baseURL string
	client  *http.Client
	apiKey  string
}

// Option configures a ServiceTool instance.
type Option func(*ServiceTool)

// WithHTTPClient injects a custom HTTP client (default 30s timeout).
func WithHTTPClient(client *http.Client) Option {
	return func(t *ServiceTool) {
		if client != nil {
			t.client = client
		}
	}
}

// WithAPIKey attaches a bearer token or API key sent as Authorization header.
func WithAPIKey(key string) Option {
	return func(t *ServiceTool) {
		t.apiKey = strings.TrimSpace(key)
	}
}

// WithTimeout sets the HTTP client timeout.
func WithTimeout(timeout time.Duration) Option {
	return func(t *ServiceTool) {
		if timeout > 0 {
			if t.client == nil {
				t.client = &http.Client{Timeout: timeout}
			} else {
				t.client.Timeout = timeout
			}
		}
	}
}

// NewServiceTool creates a tool targeting the extract-service base URL (e.g. http://extract-service:8081).
func NewServiceTool(baseURL string, opts ...Option) *ServiceTool {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	tool := &ServiceTool{
		baseURL: trimmed,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(tool)
	}
	return tool
}

// Name satisfies the tools.Tool interface.
func (t *ServiceTool) Name() string {
	return "extract_service"
}

// Description provides guidance to the LLM on how to use the tool.
func (t *ServiceTool) Description() string {
	return "Call the Layer 4 extract service. Pass JSON input with an \"action\" field (extract, training). " +
		"For action=\"extract\" include request fields accepted by the /extract endpoint. " +
		"For action=\"training\" include a payload compatible with /generate/training (e.g. mode=document with document.inputs)."
}

// Call executes the desired endpoint and returns the raw JSON body.
func (t *ServiceTool) Call(ctx context.Context, input string) (string, error) {
	if strings.TrimSpace(input) == "" {
		return "", fmt.Errorf("input cannot be empty; provide JSON payload")
	}

	var payload map[string]any
	if err := json.Unmarshal([]byte(input), &payload); err != nil {
		return "", fmt.Errorf("failed to parse JSON input: %w", err)
	}

	action := strings.ToLower(strings.TrimSpace(extractString(payload, "action")))
	delete(payload, "action")

	var endpoint string
	switch action {
	case "", "extract":
		endpoint = "/extract"
	case "training", "generate_training", "training_generation":
		endpoint = "/generate/training"
	default:
		return "", fmt.Errorf("unsupported action %q", action)
	}

	if nested, ok := payload["payload"].(map[string]any); ok {
		payload = nested
	}

	if endpoint == "/generate/training" {
		if _, ok := payload["mode"]; !ok {
			payload["mode"] = "document"
		}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.baseURL+endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if t.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", t.apiKey))
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("extract service returned %s: %s", resp.Status, strings.TrimSpace(string(respBody)))
	}

	return string(respBody), nil
}

func extractString(m map[string]any, key string) string {
	val, ok := m[key]
	if !ok {
		return ""
	}
	switch v := val.(type) {
	case string:
		return v
	default:
		return fmt.Sprintf("%v", v)
	}
}
