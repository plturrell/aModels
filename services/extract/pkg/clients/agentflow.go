package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// AgentFlowClient provides direct HTTP integration with AgentFlow service
type AgentFlowClient struct {
	baseURL    string
	httpClient *http.Client
	logger     *log.Logger
}

// AgentFlowRunRequest represents a request to run an AgentFlow flow
type AgentFlowRunRequest struct {
	FlowID     string                 `json:"flow_id"`
	InputValue string                 `json:"input_value,omitempty"`
	Inputs     map[string]any         `json:"inputs,omitempty"`
	Ensure     bool                   `json:"ensure,omitempty"`
}

// AgentFlowRunResponse represents the response from AgentFlow flow execution
type AgentFlowRunResponse struct {
	LocalID  string         `json:"local_id,omitempty"`
	RemoteID string         `json:"remote_id,omitempty"`
	Result   map[string]any `json:"result,omitempty"`
	Error    string         `json:"error,omitempty"`
}

// NewAgentFlowClient creates a new AgentFlow client with connection pooling
func NewAgentFlowClient(logger *log.Logger) *AgentFlowClient {
	baseURL := os.Getenv("AGENTFLOW_SERVICE_URL")
	if baseURL == "" {
		baseURL = "http://agentflow-service:9001"
	}

	// Create HTTP client with connection pooling
	maxIdleConns := 10
	if val := os.Getenv("AGENTFLOW_HTTP_POOL_SIZE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxIdleConns = parsed
		}
	}

	maxIdleConnsPerHost := 5
	if val := os.Getenv("AGENTFLOW_HTTP_MAX_IDLE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxIdleConnsPerHost = parsed
		}
	}

	transport := &http.Transport{
		MaxIdleConns:        maxIdleConns,
		MaxIdleConnsPerHost: maxIdleConnsPerHost,
		IdleConnTimeout:     90 * time.Second,
		DisableKeepAlives:   false,
	}

	httpClient := &http.Client{
		Transport: transport,
		Timeout:   120 * time.Second,
	}

	if logger != nil {
		logger.Printf("AgentFlow client initialized: %s (with connection pooling)", baseURL)
	}

	return &AgentFlowClient{
		baseURL:    baseURL,
		httpClient: httpClient,
		logger:     logger,
	}
}

// RunFlow executes an AgentFlow flow directly
func (c *AgentFlowClient) RunFlow(ctx context.Context, req *AgentFlowRunRequest) (*AgentFlowRunResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if req.FlowID == "" {
		return nil, fmt.Errorf("flow_id is required")
	}

	endpoint := fmt.Sprintf("%s/flows/%s/run", strings.TrimSuffix(c.baseURL, "/"), req.FlowID)

	body, err := json.Marshal(map[string]any{
		"input_value": req.InputValue,
		"inputs":      req.Inputs,
		"ensure":      req.Ensure,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Get retry configuration
	maxRetries := 3
	if val := os.Getenv("AGENTFLOW_RETRY_MAX_ATTEMPTS"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxRetries = parsed
		}
	}

	initialBackoff := 100 * time.Millisecond
	maxBackoff := 2 * time.Second

	var resp *http.Response
	retryErr := retryWithExponentialBackoff(ctx, c.logger, maxRetries, initialBackoff, maxBackoff, func() error {
		var err error
		resp, err = c.httpClient.Do(httpReq)
		if err != nil {
			if isRetryableError(err) {
				return err
			}
			return fmt.Errorf("non-retryable error: %w", err)
		}
		return nil
	})

	if retryErr != nil {
		return nil, fmt.Errorf("AgentFlow request failed: %w", retryErr)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("AgentFlow request failed: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	var flowResp AgentFlowRunResponse
	if err := json.NewDecoder(resp.Body).Decode(&flowResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &flowResp, nil
}

// HealthCheck checks if AgentFlow service is available
func (c *AgentFlowClient) HealthCheck(ctx context.Context) error {
	endpoint := fmt.Sprintf("%s/healthz", strings.TrimSuffix(c.baseURL, "/"))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: status=%d", resp.StatusCode)
	}

	return nil
}

// ConvertAndRunWorkflow converts a workflow to AgentFlow format and runs it
func (c *AgentFlowClient) ConvertAndRunWorkflow(
	ctx context.Context,
	workflow *AgentFlowWorkflow,
	inputValue string,
	inputs map[string]any,
) (*AgentFlowRunResponse, error) {
	// First, ensure the flow exists by importing it
	// This would require an import endpoint in AgentFlow, but for now
	// we'll use the flow_id directly if it exists

	// For now, we'll use a temporary flow_id based on workflow name
	flowID := fmt.Sprintf("extract/%s", workflow.Name)
	if workflow.Metadata != nil {
		if id, ok := workflow.Metadata["id"].(string); ok && id != "" {
			flowID = id
		}
	}

	req := &AgentFlowRunRequest{
		FlowID:     flowID,
		InputValue: inputValue,
		Inputs:     inputs,
		Ensure:     true, // Ensure flow exists
	}

	return c.RunFlow(ctx, req)
}

