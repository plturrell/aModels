package httpclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
)

// CircuitBreakerState represents the state of a circuit breaker.
type CircuitBreakerState int

const (
	CircuitBreakerClosed CircuitBreakerState = iota
	CircuitBreakerOpen
	CircuitBreakerHalfOpen
)

// CircuitBreaker implements the circuit breaker pattern.
type CircuitBreaker struct {
	maxFailures      int
	resetTimeout     time.Duration
	state            CircuitBreakerState
	failureCount     int
	lastFailureTime  time.Time
	mu               sync.RWMutex
	onStateChange    func(state CircuitBreakerState)
}

// NewCircuitBreaker creates a new circuit breaker.
func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures:  maxFailures,
		resetTimeout: resetTimeout,
		state:        CircuitBreakerClosed,
	}
}

// OnStateChange sets a callback for state changes.
func (cb *CircuitBreaker) OnStateChange(fn func(state CircuitBreakerState)) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.onStateChange = fn
}

// Call executes a function with circuit breaker protection.
func (cb *CircuitBreaker) Call(fn func() error) error {
	cb.mu.Lock()
	
	// Check if we should attempt to reset
	if cb.state == CircuitBreakerOpen {
		if time.Since(cb.lastFailureTime) >= cb.resetTimeout {
			cb.state = CircuitBreakerHalfOpen
			cb.failureCount = 0
			cb.notifyStateChange()
		} else {
			cb.mu.Unlock()
			return fmt.Errorf("circuit breaker is open")
		}
	}
	cb.mu.Unlock()

	// Execute the function
	err := fn()

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		cb.failureCount++
		cb.lastFailureTime = time.Now()

		if cb.failureCount >= cb.maxFailures {
			if cb.state != CircuitBreakerOpen {
				cb.state = CircuitBreakerOpen
				cb.notifyStateChange()
			}
		} else if cb.state == CircuitBreakerHalfOpen {
			// Half-open failed, go back to open
			cb.state = CircuitBreakerOpen
			cb.notifyStateChange()
		}
		return err
	}

	// Success - reset if we were in half-open
	if cb.state == CircuitBreakerHalfOpen {
		cb.state = CircuitBreakerClosed
		cb.failureCount = 0
		cb.notifyStateChange()
	} else if cb.state == CircuitBreakerClosed {
		// Reset failure count on success
		cb.failureCount = 0
	}

	return nil
}

// State returns the current circuit breaker state.
func (cb *CircuitBreaker) State() CircuitBreakerState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

func (cb *CircuitBreaker) notifyStateChange() {
	if cb.onStateChange != nil {
		go cb.onStateChange(cb.state)
	}
}

// RetryConfig configures retry behavior.
type RetryConfig struct {
	MaxAttempts      int
	InitialBackoff   time.Duration
	MaxBackoff       time.Duration
	BackoffMultiplier float64
	RetryableStatusCodes []int // HTTP status codes that should trigger retry
}

// DefaultRetryConfig returns a default retry configuration.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:        3,
		InitialBackoff:    100 * time.Millisecond,
		MaxBackoff:         5 * time.Second,
		BackoffMultiplier: 2.0,
		RetryableStatusCodes: []int{
			http.StatusInternalServerError,
			http.StatusBadGateway,
			http.StatusServiceUnavailable,
			http.StatusGatewayTimeout,
		},
	}
}

// IsRetryableStatusCode checks if a status code should trigger a retry.
func (rc RetryConfig) IsRetryableStatusCode(code int) bool {
	for _, retryable := range rc.RetryableStatusCodes {
		if code == retryable {
			return true
		}
	}
	return false
}

// Client provides an HTTP client with retry logic and circuit breaker.
type Client struct {
	httpClient     *http.Client
	circuitBreaker *CircuitBreaker
	retryConfig    RetryConfig
	logger         *log.Logger
}

// ClientConfig configures the HTTP client.
type ClientConfig struct {
	Timeout         time.Duration
	MaxRetries      int
	InitialBackoff  time.Duration
	MaxBackoff      time.Duration
	CircuitBreaker  *CircuitBreaker
	Logger          *log.Logger
}

// NewClient creates a new HTTP client with retry and circuit breaker support.
func NewClient(config ClientConfig) *Client {
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	retryConfig := DefaultRetryConfig()
	if config.MaxRetries > 0 {
		retryConfig.MaxAttempts = config.MaxRetries
	}
	if config.InitialBackoff > 0 {
		retryConfig.InitialBackoff = config.InitialBackoff
	}
	if config.MaxBackoff > 0 {
		retryConfig.MaxBackoff = config.MaxBackoff
	}

	if config.CircuitBreaker == nil {
		config.CircuitBreaker = NewCircuitBreaker(5, 30*time.Second)
	}

	return &Client{
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		circuitBreaker: config.CircuitBreaker,
		retryConfig:    retryConfig,
		logger:         config.Logger,
	}
}

// Do executes an HTTP request with retry logic and circuit breaker protection.
func (c *Client) Do(req *http.Request) (*http.Response, error) {
	var lastErr error
	backoff := c.retryConfig.InitialBackoff

	for attempt := 0; attempt < c.retryConfig.MaxAttempts; attempt++ {
		if attempt > 0 {
			// Wait before retry
			select {
			case <-req.Context().Done():
				return nil, req.Context().Err()
			case <-time.After(backoff):
			}

			// Exponential backoff
			backoff = time.Duration(float64(backoff) * c.retryConfig.BackoffMultiplier)
			if backoff > c.retryConfig.MaxBackoff {
				backoff = c.retryConfig.MaxBackoff
			}

			if c.logger != nil {
				c.logger.Printf("Retrying HTTP request (attempt %d/%d): %s %s", 
					attempt+1, c.retryConfig.MaxAttempts, req.Method, req.URL.Path)
			}
		}

		// Check circuit breaker
		var resp *http.Response
		err := c.circuitBreaker.Call(func() error {
			var callErr error
			resp, callErr = c.httpClient.Do(req)
			if callErr != nil {
				return callErr
			}

			// Check if status code is retryable
			if c.retryConfig.IsRetryableStatusCode(resp.StatusCode) {
				resp.Body.Close()
				return fmt.Errorf("retryable status code: %d", resp.StatusCode)
			}

			return nil
		})

		if err != nil {
			// Circuit breaker is open or request failed
			if err.Error() == "circuit breaker is open" {
				if c.logger != nil {
					c.logger.Printf("Circuit breaker is open, skipping request: %s %s", req.Method, req.URL.Path)
				}
				return nil, fmt.Errorf("circuit breaker is open: %w", err)
			}

			lastErr = err
			// If this is the last attempt, return the error
			if attempt == c.retryConfig.MaxAttempts-1 {
				return nil, fmt.Errorf("request failed after %d attempts: %w", c.retryConfig.MaxAttempts, err)
			}
			continue
		}

		// Success
		if attempt > 0 && c.logger != nil {
			c.logger.Printf("Request succeeded after %d attempts: %s %s", attempt+1, req.Method, req.URL.Path)
		}
		return resp, nil
	}

	return nil, fmt.Errorf("request failed after %d attempts: %w", c.retryConfig.MaxAttempts, lastErr)
}

// DoJSON executes an HTTP request and unmarshals the JSON response.
func (c *Client) DoJSON(req *http.Request, target interface{}) error {
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Read response body for error details
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	if target != nil {
		if err := json.NewDecoder(resp.Body).Decode(target); err != nil {
			return fmt.Errorf("failed to decode JSON response: %w", err)
		}
	}

	return nil
}

// PostJSON sends a POST request with JSON body and unmarshals the response.
func (c *Client) PostJSON(ctx context.Context, url string, body interface{}, target interface{}) error {
	var bodyReader io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bodyReader)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	return c.DoJSON(req, target)
}

// GetJSON sends a GET request and unmarshals the JSON response.
func (c *Client) GetJSON(ctx context.Context, url string, target interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	return c.DoJSON(req, target)
}

