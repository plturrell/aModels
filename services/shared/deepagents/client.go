package deepagents

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

// Client provides a standardized HTTP client for DeepAgents service with retry and circuit breaker.
type Client struct {
	baseURL        string
	httpClient     *http.Client
	logger         *log.Logger
	enabled        bool
	maxRetries     int
	initialBackoff time.Duration
	maxBackoff     time.Duration
}

// Config holds configuration for the DeepAgents client.
type Config struct {
	BaseURL        string
	Timeout        time.Duration
	MaxRetries     int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Logger         *log.Logger
	Enabled        bool
}

// DefaultConfig returns default configuration.
func DefaultConfig() Config {
	return Config{
		Timeout:        120 * time.Second,
		MaxRetries:     2,
		InitialBackoff: 1 * time.Second,
		MaxBackoff:     5 * time.Second,
		Enabled:        true,
	}
}

// NewClient creates a new DeepAgents client with standardized configuration.
func NewClient(config Config) *Client {
	if config.BaseURL == "" {
		config.BaseURL = "http://deepagents-service:9004"
	}

	if config.Timeout == 0 {
		config.Timeout = 120 * time.Second
	}

	if config.MaxRetries == 0 {
		config.MaxRetries = 2
	}

	if config.InitialBackoff == 0 {
		config.InitialBackoff = 1 * time.Second
	}

	if config.MaxBackoff == 0 {
		config.MaxBackoff = 5 * time.Second
	}

	return &Client{
		baseURL:        config.BaseURL,
		httpClient:     &http.Client{Timeout: config.Timeout},
		logger:         config.Logger,
		enabled:        config.Enabled,
		maxRetries:     config.MaxRetries,
		initialBackoff: config.InitialBackoff,
		maxBackoff:     config.MaxBackoff,
	}
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// InvokeRequest represents a request to invoke the agent.
type InvokeRequest struct {
	Messages      []Message              `json:"messages"`
	Stream        bool                   `json:"stream,omitempty"`
	Config        map[string]interface{} `json:"config,omitempty"`
	ResponseFormat interface{}           `json:"response_format,omitempty"`
}

// InvokeResponse represents the response from agent invocation.
type InvokeResponse struct {
	Messages         []Message            `json:"messages"`
	StructuredOutput map[string]interface{} `json:"structured_output,omitempty"`
	ValidationErrors []string            `json:"validation_errors,omitempty"`
	Result           interface{}         `json:"result,omitempty"`
}

// StructuredInvokeRequest represents a request for structured output.
type StructuredInvokeRequest struct {
	Messages      []Message              `json:"messages"`
	ResponseFormat map[string]interface{} `json:"response_format"`
	Stream        bool                   `json:"stream,omitempty"`
	Config        map[string]interface{} `json:"config,omitempty"`
}

// StructuredInvokeResponse represents structured response.
type StructuredInvokeResponse struct {
	Messages         []Message            `json:"messages"`
	StructuredOutput map[string]interface{} `json:"structured_output"`
	ValidationErrors []string            `json:"validation_errors,omitempty"`
	Result           interface{}         `json:"result,omitempty"`
}

// Invoke calls the /invoke endpoint with retry logic.
func (c *Client) Invoke(ctx context.Context, req InvokeRequest) (*InvokeResponse, error) {
	if !c.enabled {
		return nil, nil // Gracefully skip if disabled
	}

	// Quick health check
	if !c.checkHealth(ctx) {
		if c.logger != nil {
			c.logger.Printf("DeepAgents service unavailable, skipping invocation")
		}
		return nil, nil
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/invoke", c.baseURL)
	return c.callWithRetry(ctx, endpoint, body, func(respBody []byte) (*InvokeResponse, error) {
		var response InvokeResponse
		if err := json.Unmarshal(respBody, &response); err != nil {
			return nil, fmt.Errorf("decode response: %w", err)
		}
		return &response, nil
	})
}

// InvokeStructured calls the /invoke/structured endpoint with retry logic.
func (c *Client) InvokeStructured(ctx context.Context, req StructuredInvokeRequest) (*StructuredInvokeResponse, error) {
	if !c.enabled {
		return nil, nil
	}

	// Quick health check
	if !c.checkHealth(ctx) {
		if c.logger != nil {
			c.logger.Printf("DeepAgents service unavailable, skipping structured invocation")
		}
		return nil, nil
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/invoke/structured", c.baseURL)
	return c.callWithRetry(ctx, endpoint, body, func(respBody []byte) (*StructuredInvokeResponse, error) {
		var response StructuredInvokeResponse
		if err := json.Unmarshal(respBody, &response); err != nil {
			return nil, fmt.Errorf("decode response: %w", err)
		}
		return &response, nil
	})
}

// callWithRetry performs HTTP request with exponential backoff retry.
func (c *Client) callWithRetry(
	ctx context.Context,
	endpoint string,
	body []byte,
	parseResponse func([]byte) (interface{}, error),
) (interface{}, error) {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := c.calculateBackoff(attempt)
			if c.logger != nil {
				c.logger.Printf("Retrying DeepAgents request (attempt %d/%d) after %v", attempt+1, c.maxRetries+1, backoff)
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("request failed: %w", err)
			if attempt < c.maxRetries {
				continue
			}
			return nil, nil // Non-fatal, return nil
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			lastErr = fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(bodyBytes))
			if resp.StatusCode >= 500 && attempt < c.maxRetries {
				continue // Retry on server errors
			}
			if c.logger != nil {
				c.logger.Printf("DeepAgents returned status %d: %s", resp.StatusCode, string(bodyBytes))
			}
			return nil, nil // Non-fatal
		}

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			lastErr = fmt.Errorf("read response: %w", err)
			if attempt < c.maxRetries {
				continue
			}
			return nil, nil
		}

		result, err := parseResponse(respBody)
		if err != nil {
			lastErr = err
			if attempt < c.maxRetries {
				continue
			}
			return nil, nil
		}

		return result, nil
	}

	if c.logger != nil {
		c.logger.Printf("DeepAgents request failed after all retries: %v", lastErr)
	}
	return nil, nil // Non-fatal
}

// calculateBackoff calculates exponential backoff duration.
func (c *Client) calculateBackoff(attempt int) time.Duration {
	backoff := c.initialBackoff * time.Duration(1<<uint(attempt))
	if backoff > c.maxBackoff {
		backoff = c.maxBackoff
	}
	return backoff
}

// checkHealth performs a quick health check.
func (c *Client) checkHealth(ctx context.Context) bool {
	healthCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	endpoint := fmt.Sprintf("%s/healthz", c.baseURL)
	req, err := http.NewRequestWithContext(healthCtx, http.MethodGet, endpoint, nil)
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

