package transformers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// CircuitBreakerState represents the state of the circuit breaker
type CircuitBreakerState int

const (
	CircuitBreakerClosed CircuitBreakerState = iota // Normal operation
	CircuitBreakerOpen                               // Circuit is open, requests fail fast
	CircuitBreakerHalfOpen                           // Testing if service recovered
)

// Client wraps an OpenAI-compatible chat completion endpoint.
type Client struct {
	endpoint   string
	modelName  string
	httpClient *http.Client
	transport  *http.Transport
	maxRetries int
	retryDelay time.Duration
	
	// Circuit breaker fields
	cbState          CircuitBreakerState
	cbMu             sync.RWMutex
	cbFailureCount   int
	cbSuccessCount   int
	cbLastFailure    time.Time
	cbOpenTimeout    time.Duration
	cbFailureThreshold int
	cbSuccessThreshold int
	healthCheckURL   string
}

// NewClient creates a new transformers client with connection pooling and retry logic
func NewClient(endpoint, modelName string, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}

	// Create transport with connection pooling
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:    90 * time.Second,
	}

	maxRetries := 3
	if envRetries := os.Getenv("TRANSFORMERS_RETRY_MAX"); envRetries != "" {
		if parsed, err := strconv.Atoi(envRetries); err == nil && parsed > 0 {
			maxRetries = parsed
		}
	}

	// Circuit breaker configuration
	failureThreshold := 5
	if envThreshold := os.Getenv("TRANSFORMERS_CB_FAILURE_THRESHOLD"); envThreshold != "" {
		if parsed, err := strconv.Atoi(envThreshold); err == nil && parsed > 0 {
			failureThreshold = parsed
		}
	}
	
	openTimeout := 30 * time.Second
	if envTimeout := os.Getenv("TRANSFORMERS_CB_OPEN_TIMEOUT"); envTimeout != "" {
		if parsed, err := time.ParseDuration(envTimeout); err == nil && parsed > 0 {
			openTimeout = parsed
		}
	}
	
	successThreshold := 2
	if envSuccess := os.Getenv("TRANSFORMERS_CB_SUCCESS_THRESHOLD"); envSuccess != "" {
		if parsed, err := strconv.Atoi(envSuccess); err == nil && parsed > 0 {
			successThreshold = parsed
		}
	}
	
	// Health check URL (defaults to /health if endpoint is set)
	healthCheckURL := ""
	if endpoint != "" {
		healthCheckURL = os.Getenv("TRANSFORMERS_HEALTH_CHECK_URL")
		if healthCheckURL == "" {
			// Try to construct health check URL from endpoint
			healthCheckURL = endpoint + "/health"
		}
	}

	return &Client{
		endpoint:  endpoint,
		modelName: modelName,
		transport: transport,
		httpClient: &http.Client{
			Timeout:   timeout,
			Transport: transport,
		},
		maxRetries: maxRetries,
		retryDelay: 1 * time.Second,
		cbState: CircuitBreakerClosed,
		cbFailureCount: 0,
		cbSuccessCount: 0,
		cbOpenTimeout: openTimeout,
		cbFailureThreshold: failureThreshold,
		cbSuccessThreshold: successThreshold,
		healthCheckURL: healthCheckURL,
	}
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Generate sends the messages to the configured chat completions endpoint and returns the text + token usage.
// It includes retry logic with exponential backoff and circuit breaker protection.
func (c *Client) Generate(ctx context.Context, messages []ChatMessage, maxTokens int, temperature, topP float64) (string, int, error) {
	if c == nil {
		return "", 0, fmt.Errorf("transformers client is nil")
	}
	if len(messages) == 0 {
		return "", 0, fmt.Errorf("messages cannot be empty")
	}
	if maxTokens <= 0 {
		maxTokens = 128
	}
	if temperature < 0 {
		temperature = 0
	}
	if topP <= 0 || topP > 1 {
		topP = 1
	}

	// Check circuit breaker state
	if !c.canProceed() {
		return "", 0, fmt.Errorf("circuit breaker is open: transformers service unavailable")
	}

	payload := chatCompletionRequest{
		Model:       c.modelName,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        topP,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", 0, fmt.Errorf("marshal request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 1s, 2s, 4s
			backoff := time.Duration(math.Pow(2, float64(attempt-1))) * c.retryDelay
			select {
			case <-ctx.Done():
				return "", 0, ctx.Err()
			case <-time.After(backoff):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
		if err != nil {
			return "", 0, fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("send request: %w", err)
			continue // Retry on network errors
		}

		if resp.StatusCode != http.StatusOK {
			data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			resp.Body.Close()
			// Don't retry on client errors (4xx), only on server errors (5xx)
			if resp.StatusCode >= 400 && resp.StatusCode < 500 {
				return "", 0, fmt.Errorf("transformers backend status %d: %s", resp.StatusCode, data)
			}
			lastErr = fmt.Errorf("transformers backend status %d: %s", resp.StatusCode, data)
			continue // Retry on server errors
		}

		var ccResp chatCompletionResponse
		if err := json.NewDecoder(resp.Body).Decode(&ccResp); err != nil {
			resp.Body.Close()
			lastErr = fmt.Errorf("decode response: %w", err)
			continue // Retry on decode errors
		}
		resp.Body.Close()

		if len(ccResp.Choices) == 0 {
			c.recordFailure()
			return "", 0, fmt.Errorf("transformers backend returned no choices")
		}

		content := ccResp.Choices[0].Message.Content
		c.recordSuccess()
		return content, ccResp.Usage.TotalTokens, nil
	}

	c.recordFailure()
	return "", 0, fmt.Errorf("transformers backend failed after %d retries: %w", c.maxRetries, lastErr)
}

// canProceed checks if the circuit breaker allows the request to proceed
func (c *Client) canProceed() bool {
	c.cbMu.Lock()
	defer c.cbMu.Unlock()

	switch c.cbState {
	case CircuitBreakerClosed:
		return true
	case CircuitBreakerOpen:
		// Check if timeout has passed, transition to half-open
		if time.Since(c.cbLastFailure) >= c.cbOpenTimeout {
			c.cbState = CircuitBreakerHalfOpen
			c.cbSuccessCount = 0
			return true
		}
		return false
	case CircuitBreakerHalfOpen:
		return true
	default:
		return true
	}
}

// recordSuccess records a successful request and updates circuit breaker state
func (c *Client) recordSuccess() {
	c.cbMu.Lock()
	defer c.cbMu.Unlock()

	switch c.cbState {
	case CircuitBreakerClosed:
		// Reset failure count on success
		c.cbFailureCount = 0
	case CircuitBreakerHalfOpen:
		c.cbSuccessCount++
		if c.cbSuccessCount >= c.cbSuccessThreshold {
			// Service recovered, close the circuit
			c.cbState = CircuitBreakerClosed
			c.cbFailureCount = 0
			c.cbSuccessCount = 0
		}
	}
}

// recordFailure records a failed request and updates circuit breaker state
func (c *Client) recordFailure() {
	c.cbMu.Lock()
	defer c.cbMu.Unlock()

	c.cbFailureCount++
	c.cbLastFailure = time.Now()

	switch c.cbState {
	case CircuitBreakerClosed:
		if c.cbFailureCount >= c.cbFailureThreshold {
			// Too many failures, open the circuit
			c.cbState = CircuitBreakerOpen
		}
	case CircuitBreakerHalfOpen:
		// Failure in half-open state, immediately open again
		c.cbState = CircuitBreakerOpen
		c.cbSuccessCount = 0
	}
}

// HealthCheck performs a health check on the transformers service
func (c *Client) HealthCheck(ctx context.Context) error {
	if c.healthCheckURL == "" {
		// No health check URL configured, skip
		return nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.healthCheckURL, nil)
	if err != nil {
		return fmt.Errorf("create health check request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		c.recordFailure()
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.recordFailure()
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	c.recordSuccess()
	return nil
}

// GetCircuitBreakerState returns the current circuit breaker state
func (c *Client) GetCircuitBreakerState() CircuitBreakerState {
	c.cbMu.RLock()
	defer c.cbMu.RUnlock()
	return c.cbState
}

// Close closes the HTTP transport and releases resources
func (c *Client) Close() {
	if c.transport != nil {
		c.transport.CloseIdleConnections()
	}
}
