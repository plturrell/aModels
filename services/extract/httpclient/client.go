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

// CorrelationIDKey is the context key for correlation ID.
const CorrelationIDKey = "correlation_id"

// AuthTokenKey is the context key for auth token.
const AuthTokenKey = "auth_token"

// RequestIDHeader is the HTTP header name for correlation ID.
const RequestIDHeader = "X-Request-ID"

// TraceIDHeader is the HTTP header name for trace ID.
const TraceIDHeader = "X-Trace-ID"

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
	httpClient          *http.Client
	circuitBreaker      *CircuitBreaker
	retryConfig         RetryConfig
	logger              *log.Logger
	baseURL             string
	healthCheckPath     string
	healthCheckCacheTTL time.Duration
	healthCheckCache    *healthCheckCache
	metricsCollector    MetricsCollector
	healthChecker       HealthChecker
}

type healthCheckCache struct {
	mu        sync.RWMutex
	isHealthy bool
	cachedAt  time.Time
}

// MetricsCollector is a function that collects metrics for HTTP requests.
type MetricsCollector func(service, endpoint string, statusCode int, latency time.Duration, correlationID string)

// HealthChecker is a function that checks service health.
type HealthChecker func(ctx context.Context, baseURL string) (bool, error)

// ResponseValidator is a function that validates response structure.
type ResponseValidator func(data map[string]interface{}) error

// ClientConfig configures the HTTP client.
type ClientConfig struct {
	Timeout            time.Duration
	MaxRetries         int
	InitialBackoff     time.Duration
	MaxBackoff         time.Duration
	CircuitBreaker     *CircuitBreaker
	Logger             *log.Logger
	BaseURL            string
	HealthCheckPath    string
	HealthCheckCacheTTL time.Duration
	MetricsCollector   MetricsCollector
	HealthChecker      HealthChecker
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

	if config.HealthCheckPath == "" {
		config.HealthCheckPath = "/healthz"
	}

	if config.HealthCheckCacheTTL == 0 {
		config.HealthCheckCacheTTL = 30 * time.Second
	}

	return &Client{
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		circuitBreaker:      config.CircuitBreaker,
		retryConfig:         retryConfig,
		logger:              config.Logger,
		baseURL:             config.BaseURL,
		healthCheckPath:     config.HealthCheckPath,
		healthCheckCacheTTL: config.HealthCheckCacheTTL,
		healthCheckCache:    &healthCheckCache{},
		metricsCollector:    config.MetricsCollector,
		healthChecker:       config.HealthChecker,
	}
}

// getCorrelationID extracts correlation ID from context or generates a new one.
func getCorrelationID(ctx context.Context) string {
	// Try to get from context
	if correlationID, ok := ctx.Value(CorrelationIDKey).(string); ok && correlationID != "" {
		return correlationID
	}
	// Try to get from request header (if context has request)
	if req, ok := ctx.Value("http_request").(*http.Request); ok {
		if correlationID := req.Header.Get(RequestIDHeader); correlationID != "" {
			return correlationID
		}
		if traceID := req.Header.Get(TraceIDHeader); traceID != "" {
			return traceID
		}
	}
	// Generate new one
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// getAuthToken extracts auth token from context.
func getAuthToken(ctx context.Context) string {
	if token, ok := ctx.Value(AuthTokenKey).(string); ok {
		return token
	}
	// Try to get from request header
	if req, ok := ctx.Value("http_request").(*http.Request); ok {
		if authHeader := req.Header.Get("Authorization"); authHeader != "" {
			return authHeader
		}
	}
	return ""
}

// checkHealth checks service health with caching.
func (c *Client) checkHealth(ctx context.Context) (bool, error) {
	now := time.Now()
	
	// Check cache
	c.healthCheckCache.mu.RLock()
	if c.healthCheckCache.cachedAt.Add(c.healthCheckCacheTTL).After(now) {
		isHealthy := c.healthCheckCache.isHealthy
		c.healthCheckCache.mu.RUnlock()
		return isHealthy, nil
	}
	c.healthCheckCache.mu.RUnlock()
	
	// Perform health check
	var isHealthy bool
	var err error
	
	if c.healthChecker != nil {
		isHealthy, err = c.healthChecker(ctx, c.baseURL)
	} else {
		// Default health check
		healthURL := c.baseURL + c.healthCheckPath
		req, reqErr := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
		if reqErr != nil {
			return false, reqErr
		}
		
		resp, respErr := c.httpClient.Do(req)
		if respErr != nil {
			isHealthy = false
			err = respErr
		} else {
			resp.Body.Close()
			isHealthy = resp.StatusCode == 200
		}
	}
	
	// Update cache
	c.healthCheckCache.mu.Lock()
	c.healthCheckCache.isHealthy = isHealthy
	c.healthCheckCache.cachedAt = now
	c.healthCheckCache.mu.Unlock()
	
	if err != nil && c.logger != nil {
		c.logger.Printf("Health check failed for %s: %v", c.baseURL, err)
	}
	
	return isHealthy, err
}

// Do executes an HTTP request with retry logic and circuit breaker protection.
func (c *Client) Do(req *http.Request) (*http.Response, error) {
	ctx := req.Context()
	correlationID := getCorrelationID(ctx)
	
	// Add correlation ID to headers if not present
	if req.Header.Get(RequestIDHeader) == "" {
		req.Header.Set(RequestIDHeader, correlationID)
	}
	
	// Add auth token if available
	if authToken := getAuthToken(ctx); authToken != "" {
		if req.Header.Get("Authorization") == "" {
			req.Header.Set("Authorization", authToken)
		}
	}
	
	// Check health before request
	if c.baseURL != "" {
		isHealthy, err := c.checkHealth(ctx)
		if err != nil || !isHealthy {
			serviceName := c.baseURL
			if c.metricsCollector != nil {
				c.metricsCollector(serviceName, req.URL.Path, 503, 0, correlationID)
			}
			return nil, NewServiceUnavailableError(serviceName, correlationID, err)
		}
	}
	
	var lastErr error
	backoff := c.retryConfig.InitialBackoff
	startTime := time.Now()
	serviceName := c.baseURL
	if serviceName == "" {
		serviceName = req.URL.Host
	}
	endpoint := req.URL.Path

	for attempt := 0; attempt < c.retryConfig.MaxAttempts; attempt++ {
		if attempt > 0 {
			// Wait before retry
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}

			// Exponential backoff
			backoff = time.Duration(float64(backoff) * c.retryConfig.BackoffMultiplier)
			if backoff > c.retryConfig.MaxBackoff {
				backoff = c.retryConfig.MaxBackoff
			}

			if c.logger != nil {
				c.logger.Printf("[%s] Retrying HTTP request (attempt %d/%d): %s %s", 
					correlationID, attempt+1, c.retryConfig.MaxAttempts, req.Method, req.URL.Path)
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

		latency := time.Since(startTime)
		
		if err != nil {
			// Circuit breaker is open or request failed
			if err.Error() == "circuit breaker is open" {
				if c.logger != nil {
					c.logger.Printf("[%s] Circuit breaker is open, skipping request: %s %s", 
						correlationID, req.Method, req.URL.Path)
				}
				if c.metricsCollector != nil {
					c.metricsCollector(serviceName, endpoint, 503, latency, correlationID)
				}
				return nil, NewServiceUnavailableError(serviceName, correlationID, err)
			}

			lastErr = err
			statusCode := 500
			if resp != nil {
				statusCode = resp.StatusCode
			}
			
			if c.metricsCollector != nil {
				c.metricsCollector(serviceName, endpoint, statusCode, latency, correlationID)
			}
			
			// If this is the last attempt, return the error
			if attempt == c.retryConfig.MaxAttempts-1 {
				return nil, &IntegrationError{
					Message:       fmt.Sprintf("request failed after %d attempts", c.retryConfig.MaxAttempts),
					Service:       serviceName,
					CorrelationID: correlationID,
					StatusCode:    statusCode,
					Err:           err,
				}
			}
			continue
		}

		// Success
		if attempt > 0 && c.logger != nil {
			c.logger.Printf("[%s] Request succeeded after %d attempts: %s %s", 
				correlationID, attempt+1, req.Method, req.URL.Path)
		}
		
		if c.metricsCollector != nil {
			c.metricsCollector(serviceName, endpoint, resp.StatusCode, latency, correlationID)
		}
		
		if c.logger != nil {
			c.logger.Printf("[%s] %s %s -> %d (latency: %v)", 
				correlationID, req.Method, req.URL.Path, resp.StatusCode, latency)
		}
		
		return resp, nil
	}

	return nil, &IntegrationError{
		Message:       fmt.Sprintf("request failed after %d attempts", c.retryConfig.MaxAttempts),
		Service:       serviceName,
		CorrelationID: correlationID,
		Err:           lastErr,
	}
}

// DoJSON executes an HTTP request and unmarshals the JSON response.
func (c *Client) DoJSON(req *http.Request, target interface{}, validator ...ResponseValidator) error {
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Read response body for error details
		body, _ := io.ReadAll(resp.Body)
		correlationID := getCorrelationID(req.Context())
		serviceName := c.baseURL
		if serviceName == "" {
			serviceName = req.URL.Host
		}
		
		if resp.StatusCode == 401 || resp.StatusCode == 403 {
			return NewAuthenticationError(serviceName, correlationID, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body)))
		}
		
		return &IntegrationError{
			Message:       fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(body)),
			Service:       serviceName,
			CorrelationID: correlationID,
			StatusCode:    resp.StatusCode,
		}
	}

	if target != nil {
		if err := json.NewDecoder(resp.Body).Decode(target); err != nil {
			correlationID := getCorrelationID(req.Context())
			serviceName := c.baseURL
			if serviceName == "" {
				serviceName = req.URL.Host
			}
			return NewValidationError(serviceName, correlationID, fmt.Errorf("failed to decode JSON response: %w", err))
		}
		
		// Run validator if provided
		if len(validator) > 0 && validator[0] != nil {
			if targetMap, ok := target.(*map[string]interface{}); ok {
				if err := validator[0](*targetMap); err != nil {
					correlationID := getCorrelationID(req.Context())
					serviceName := c.baseURL
					if serviceName == "" {
						serviceName = req.URL.Host
					}
					return NewValidationError(serviceName, correlationID, err)
				}
			}
		}
	}

	return nil
}

// PostJSON sends a POST request with JSON body and unmarshals the response.
func (c *Client) PostJSON(ctx context.Context, url string, body interface{}, target interface{}, validator ...ResponseValidator) error {
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

	return c.DoJSON(req, target, validator...)
}

// GetJSON sends a GET request and unmarshals the JSON response.
func (c *Client) GetJSON(ctx context.Context, url string, target interface{}, validator ...ResponseValidator) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	return c.DoJSON(req, target, validator...)
}

