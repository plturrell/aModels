package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// LocalAIClient provides a standardized client for LocalAI interactions
// with explicit model selection, retry logic, and model validation.
type LocalAIClient struct {
	baseURL        string
	httpClient     *http.Client
	logger         *log.Logger
	circuitBreaker *CircuitBreaker
	modelCache     *sync.Map // Cache for model availability checks
}

// CircuitBreaker implements circuit breaker pattern for LocalAI calls
type CircuitBreaker struct {
	failures      int
	lastFailTime  time.Time
	state         string // "closed", "open", "half-open"
	mu            sync.RWMutex
	failureThreshold int
	timeout          time.Duration
}

const (
	circuitStateClosed   = "closed"
	circuitStateOpen     = "open"
	circuitStateHalfOpen = "half-open"
)

// NewLocalAIClient creates a new LocalAI client with retry logic and circuit breaker
func NewLocalAIClient(baseURL string, httpClient *http.Client, logger *log.Logger) *LocalAIClient {
	if httpClient == nil {
		// Default HTTP client with connection pooling
		transport := &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
			MaxConnsPerHost:     50,
		}
		httpClient = &http.Client{
			Transport: transport,
			Timeout:   120 * time.Second,
		}
	}

	return &LocalAIClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: httpClient,
		logger:     logger,
		circuitBreaker: &CircuitBreaker{
			failureThreshold: 5,
			timeout:         30 * time.Second,
			state:           circuitStateClosed,
		},
		modelCache: &sync.Map{},
	}
}

// StoreDocument stores a document in LocalAI with explicit model/domain selection
func (c *LocalAIClient) StoreDocument(ctx context.Context, domain, model string, payload map[string]interface{}) (map[string]interface{}, error) {
	// Validate model/domain availability
	if err := c.validateModel(ctx, domain, model); err != nil {
		if c.logger != nil {
			c.logger.Printf("Model validation failed for domain=%s, model=%s: %v", domain, model, err)
		}
		// Continue anyway - LocalAI will handle fallback
	}

	// Ensure domain and model are in payload
	if payload == nil {
		payload = make(map[string]interface{})
	}
	if domain != "" {
		payload["domain"] = domain
	}
	if model != "" {
		payload["model"] = model
	}

	// Try domain-specific endpoint first, fallback to generic
	url := c.baseURL + "/v1/documents"
	if domain != "general" && domain != "" {
		domainURL := c.baseURL + "/v1/domains/" + domain + "/documents"
		result, err := c.postWithRetry(ctx, domainURL, payload)
		if err == nil {
			return result, nil
		}
		if c.logger != nil {
			c.logger.Printf("Domain-specific storage failed, using generic endpoint: %v", err)
		}
	}

	return c.postWithRetry(ctx, url, payload)
}

// CallDomainEndpoint calls a domain-specific LocalAI endpoint
func (c *LocalAIClient) CallDomainEndpoint(ctx context.Context, domain, endpoint string, payload map[string]interface{}) (map[string]interface{}, error) {
	url := c.baseURL + "/v1/domains/" + domain + "/" + endpoint
	return c.postWithRetry(ctx, url, payload)
}

// postWithRetry performs a POST request with exponential backoff retry
func (c *LocalAIClient) postWithRetry(ctx context.Context, url string, payload map[string]interface{}) (map[string]interface{}, error) {
	// Check circuit breaker
	if !c.circuitBreaker.Allow() {
		return nil, fmt.Errorf("circuit breaker is open")
	}

	maxRetries := 3
	initialDelay := 1 * time.Second
	backoffMultiplier := 2.0

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			delay := time.Duration(float64(initialDelay) * pow(backoffMultiplier, float64(attempt-1)))
			if delay > 10*time.Second {
				delay = 10 * time.Second
			}
			if c.logger != nil {
				c.logger.Printf("Retrying LocalAI call (attempt %d/%d) after %v", attempt+1, maxRetries+1, delay)
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		result, err := c.post(ctx, url, payload)
		if err == nil {
			c.circuitBreaker.RecordSuccess()
			return result, nil
		}

		lastErr = err
		c.circuitBreaker.RecordFailure()

		// Don't retry on client errors (4xx)
		if httpErr, ok := err.(*HTTPError); ok && httpErr.StatusCode >= 400 && httpErr.StatusCode < 500 {
			break
		}
	}

	return nil, fmt.Errorf("failed after %d retries: %w", maxRetries+1, lastErr)
}

// post performs a single POST request
func (c *LocalAIClient) post(ctx context.Context, url string, payload map[string]interface{}) (map[string]interface{}, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal JSON: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &HTTPError{
			StatusCode: resp.StatusCode,
			Message:   string(body),
		}
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

// validateModel checks if a model/domain is available in LocalAI
func (c *LocalAIClient) validateModel(ctx context.Context, domain, model string) error {
	// Check cache first
	cacheKey := domain + ":" + model
	if cached, ok := c.modelCache.Load(cacheKey); ok {
		if valid, ok := cached.(bool); ok && valid {
			return nil
		}
	}

	// Try to query domain info (non-blocking, best-effort)
	// This is a lightweight check - if it fails, we'll proceed anyway
	url := c.baseURL + "/v1/domains/" + domain
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}

	// Use a short timeout for validation
	validationCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	resp, err := c.httpClient.Do(req.WithContext(validationCtx))
	if err != nil {
		// Non-fatal - cache as unavailable but continue
		c.modelCache.Store(cacheKey, false)
		return nil // Don't fail the request
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		c.modelCache.Store(cacheKey, true)
		return nil
	}

	c.modelCache.Store(cacheKey, false)
	return nil // Non-fatal
}

// HTTPError represents an HTTP error
type HTTPError struct {
	StatusCode int
	Message    string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message)
}

// Allow checks if the circuit breaker allows the request
func (cb *CircuitBreaker) Allow() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if cb.state == circuitStateClosed {
		return true
	}

	if cb.state == circuitStateOpen {
		if time.Since(cb.lastFailTime) > cb.timeout {
			cb.mu.RUnlock()
			cb.mu.Lock()
			cb.state = circuitStateHalfOpen
			cb.mu.Unlock()
			cb.mu.RLock()
			return true
		}
		return false
	}

	// Half-open state - allow one request
	return true
}

// RecordSuccess records a successful request
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.state == circuitStateHalfOpen {
		cb.state = circuitStateClosed
		cb.failures = 0
	}
}

// RecordFailure records a failed request
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failures++
	cb.lastFailTime = time.Now()

	if cb.failures >= cb.failureThreshold {
		cb.state = circuitStateOpen
	} else if cb.state == circuitStateHalfOpen {
		cb.state = circuitStateOpen
	}
}

// Helper function for power calculation
func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}

