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
// with explicit model selection, retry logic, model validation, and metrics collection.
type LocalAIClient struct {
	baseURL        string
	httpClient     *http.Client
	logger         *log.Logger
	circuitBreaker *CircuitBreaker
	modelCache     *sync.Map // Cache for model availability checks
	metrics        *LocalAIMetrics // Phase 2: Metrics collection
	timeout        time.Duration   // Phase 2: Configurable timeout
	fallbackModels map[string][]string // Phase 2: Model fallback strategy
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

// LocalAIClientConfig configures the LocalAI client (Phase 2)
type LocalAIClientConfig struct {
	BaseURL        string
	HTTPClient     *http.Client
	Logger         *log.Logger
	Timeout        time.Duration // Phase 2: Configurable timeout
	FallbackModels map[string][]string // Phase 2: Model fallback strategy
}

// NewLocalAIClient creates a new LocalAI client with retry logic and circuit breaker
func NewLocalAIClient(baseURL string, httpClient *http.Client, logger *log.Logger) *LocalAIClient {
	return NewLocalAIClientWithConfig(LocalAIClientConfig{
		BaseURL:    baseURL,
		HTTPClient: httpClient,
		Logger:     logger,
		Timeout:    120 * time.Second, // Default timeout
		FallbackModels: map[string][]string{
			// Phase 2: Model fallback strategy
			"gemma-2b-q4_k_m.gguf": {"phi-3.5-mini", "vaultgemma-1b-transformers"},
			"gemma-7b-q4_k_m.gguf": {"gemma-2b-q4_k_m.gguf", "phi-3.5-mini"},
			"granite-4.0":          {"gemma-2b-q4_k_m.gguf", "phi-3.5-mini"},
			"phi-3.5-mini":         {"vaultgemma-1b-transformers", "gemma-2b-q4_k_m.gguf"},
		},
	})
}

// NewLocalAIClientWithConfig creates a new LocalAI client with full configuration (Phase 2)
func NewLocalAIClientWithConfig(config LocalAIClientConfig) *LocalAIClient {
	httpClient := config.HTTPClient
	if httpClient == nil {
		// Default HTTP client with connection pooling
		transport := &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
			MaxConnsPerHost:     50,
		}
		timeout := config.Timeout
		if timeout == 0 {
			timeout = 120 * time.Second
		}
		httpClient = &http.Client{
			Transport: transport,
			Timeout:   timeout,
		}
	}

	fallbackModels := config.FallbackModels
	if fallbackModels == nil {
		fallbackModels = map[string][]string{
			"gemma-2b-q4_k_m.gguf": {"phi-3.5-mini", "vaultgemma-1b-transformers"},
			"gemma-7b-q4_k_m.gguf": {"gemma-2b-q4_k_m.gguf", "phi-3.5-mini"},
			"granite-4.0":          {"gemma-2b-q4_k_m.gguf", "phi-3.5-mini"},
			"phi-3.5-mini":         {"vaultgemma-1b-transformers", "gemma-2b-q4_k_m.gguf"},
		}
	}

	return &LocalAIClient{
		baseURL:    strings.TrimRight(config.BaseURL, "/"),
		httpClient: httpClient,
		logger:     config.Logger,
		circuitBreaker: &CircuitBreaker{
			failureThreshold: 5,
			timeout:         30 * time.Second,
			state:           circuitStateClosed,
		},
		modelCache:     &sync.Map{},
		metrics:        NewLocalAIMetrics(), // Phase 2: Metrics collection
		timeout:        config.Timeout,
		fallbackModels: fallbackModels, // Phase 2: Model fallback strategy
	}
}

// StoreDocument stores a document in LocalAI with explicit model/domain selection and fallback (Phase 2)
func (c *LocalAIClient) StoreDocument(ctx context.Context, domain, model string, payload map[string]interface{}) (map[string]interface{}, error) {
	startTime := time.Now()
	
	// Phase 2: Try primary model first, then fallback models
	modelsToTry := []string{model}
	if fallbacks, ok := c.fallbackModels[model]; ok {
		modelsToTry = append(modelsToTry, fallbacks...)
	}

	var lastErr error
	for _, tryModel := range modelsToTry {
		// Validate model/domain availability
		if err := c.validateModel(ctx, domain, tryModel); err != nil {
			if c.logger != nil {
				c.logger.Printf("Model validation failed for domain=%s, model=%s: %v", domain, tryModel, err)
			}
			// Continue to next model
			continue
		}

		// Ensure domain and model are in payload
		if payload == nil {
			payload = make(map[string]interface{})
		}
		if domain != "" {
			payload["domain"] = domain
		}
		if tryModel != "" {
			payload["model"] = tryModel
		}

		// Try domain-specific endpoint first, fallback to generic
		url := c.baseURL + "/v1/documents"
		if domain != "general" && domain != "" {
			domainURL := c.baseURL + "/v1/domains/" + domain + "/documents"
			result, err := c.postWithRetry(ctx, domainURL, payload)
			if err == nil {
				// Phase 2: Record metrics
				c.metrics.RecordCall("StoreDocument", domain, tryModel, time.Since(startTime), true)
				if tryModel != model && c.logger != nil {
					c.logger.Printf("Successfully used fallback model %s (original: %s) for domain %s", tryModel, model, domain)
				}
				return result, nil
			}
			lastErr = err
			if c.logger != nil {
				c.logger.Printf("Domain-specific storage failed with model %s, trying next model: %v", tryModel, err)
			}
		}

		// Try generic endpoint
		result, err := c.postWithRetry(ctx, url, payload)
		if err == nil {
			// Phase 2: Record metrics
			c.metrics.RecordCall("StoreDocument", domain, tryModel, time.Since(startTime), true)
			if tryModel != model && c.logger != nil {
				c.logger.Printf("Successfully used fallback model %s (original: %s) for domain %s", tryModel, model, domain)
			}
			return result, nil
		}
		lastErr = err
	}

	// Phase 2: Record metrics for failure
	c.metrics.RecordCall("StoreDocument", domain, model, time.Since(startTime), false)
	
	// Phase 2: Enhanced error message with context
	return nil, fmt.Errorf("failed to store document after trying %d models (domain: %s, primary model: %s): %w", len(modelsToTry), domain, model, lastErr)
}

// CallDomainEndpoint calls a domain-specific LocalAI endpoint (Phase 2: with metrics)
func (c *LocalAIClient) CallDomainEndpoint(ctx context.Context, domain, endpoint string, payload map[string]interface{}) (map[string]interface{}, error) {
	startTime := time.Now()
	url := c.baseURL + "/v1/domains/" + domain + "/" + endpoint
	
	// Extract model from payload if available
	model := ""
	if payload != nil {
		if m, ok := payload["model"].(string); ok {
			model = m
		}
	}
	
	result, err := c.postWithRetry(ctx, url, payload)
	
	// Phase 2: Record metrics
	c.metrics.RecordCall("CallDomainEndpoint_"+endpoint, domain, model, time.Since(startTime), err == nil)
	
	return result, err
}

// postWithRetry performs a POST request with exponential backoff retry (Phase 2: enhanced error messages)
func (c *LocalAIClient) postWithRetry(ctx context.Context, url string, payload map[string]interface{}) (map[string]interface{}, error) {
	// Check circuit breaker
	if !c.circuitBreaker.Allow() {
		// Phase 2: Enhanced error message
		return nil, &LocalAIError{
			Type:        "CircuitBreakerOpen",
			Message:     "circuit breaker is open - LocalAI service is unavailable",
			URL:         url,
			Retryable:   true,
			SuggestedAction: "wait for circuit breaker to close (30s timeout) or check LocalAI service health",
		}
	}

	maxRetries := 3
	initialDelay := 1 * time.Second
	backoffMultiplier := 2.0

	var lastErr error
	var lastHTTPErr *HTTPError
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
				return nil, &LocalAIError{
					Type:        "ContextCancelled",
					Message:     fmt.Sprintf("request cancelled: %v", ctx.Err()),
					URL:         url,
					Retryable:   false,
					SuggestedAction: "check request timeout settings",
				}
			case <-time.After(delay):
			}
		}

		result, err := c.post(ctx, url, payload)
		if err == nil {
			oldState := c.circuitBreaker.GetState()
			c.circuitBreaker.RecordSuccess()
			newState := c.circuitBreaker.GetState()
			// Phase 2: Record circuit breaker state change
			if oldState != newState {
				c.metrics.RecordCircuitBreakerState(newState)
			}
			return result, nil
		}

		lastErr = err
		if httpErr, ok := err.(*HTTPError); ok {
			lastHTTPErr = httpErr
		}
		oldState := c.circuitBreaker.GetState()
		c.circuitBreaker.RecordFailure()
		newState := c.circuitBreaker.GetState()
		// Phase 2: Record circuit breaker state change
		if oldState != newState {
			c.metrics.RecordCircuitBreakerState(newState)
		}

		// Don't retry on client errors (4xx)
		if httpErr, ok := err.(*HTTPError); ok && httpErr.StatusCode >= 400 && httpErr.StatusCode < 500 {
			// Phase 2: Enhanced error message for client errors
			return nil, &LocalAIError{
				Type:        "ClientError",
				Message:     fmt.Sprintf("HTTP %d: %s", httpErr.StatusCode, httpErr.Message),
				URL:         url,
				StatusCode:  httpErr.StatusCode,
				Retryable:   false,
				SuggestedAction: "check request payload and LocalAI configuration",
			}
		}
	}

	// Phase 2: Enhanced error message with context
	return nil, &LocalAIError{
		Type:        "MaxRetriesExceeded",
		Message:     fmt.Sprintf("failed after %d retries: %v", maxRetries+1, lastErr),
		URL:         url,
		Retryable:   true,
		SuggestedAction: "check LocalAI service availability and network connectivity",
		LastHTTPError: lastHTTPErr,
	}
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

// LocalAIError represents a LocalAI-specific error with enhanced context (Phase 2)
type LocalAIError struct {
	Type           string
	Message        string
	URL            string
	StatusCode     int
	Retryable      bool
	SuggestedAction string
	LastHTTPError  *HTTPError
}

func (e *LocalAIError) Error() string {
	msg := fmt.Sprintf("LocalAI error [%s]: %s (URL: %s)", e.Type, e.Message, e.URL)
	if e.StatusCode > 0 {
		msg += fmt.Sprintf(" [HTTP %d]", e.StatusCode)
	}
	if e.SuggestedAction != "" {
		msg += fmt.Sprintf(" - Suggested: %s", e.SuggestedAction)
	}
	return msg
}

// LocalAIMetrics collects metrics for LocalAI calls (Phase 2)
type LocalAIMetrics struct {
	mu                sync.RWMutex
	callCount          int64
	successCount       int64
	errorCount         int64
	totalLatency       time.Duration
	modelUsage         map[string]int64      // model -> count
	domainUsage        map[string]int64     // domain -> count
	modelLatency       map[string]time.Duration // model -> total latency
	domainLatency      map[string]time.Duration // domain -> total latency
	circuitBreakerState map[string]int64     // state -> count
	lastCallTime       time.Time
}

// NewLocalAIMetrics creates a new metrics collector
func NewLocalAIMetrics() *LocalAIMetrics {
	return &LocalAIMetrics{
		modelUsage:         make(map[string]int64),
		domainUsage:        make(map[string]int64),
		modelLatency:       make(map[string]time.Duration),
		domainLatency:      make(map[string]time.Duration),
		circuitBreakerState: make(map[string]int64),
	}
}

// RecordCall records a LocalAI call with metrics (Phase 2)
func (m *LocalAIMetrics) RecordCall(operation, domain, model string, latency time.Duration, success bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.callCount++
	m.totalLatency += latency
	m.lastCallTime = time.Now()

	if success {
		m.successCount++
	} else {
		m.errorCount++
	}

	if model != "" {
		m.modelUsage[model]++
		m.modelLatency[model] += latency
	}

	if domain != "" {
		m.domainUsage[domain]++
		m.domainLatency[domain] += latency
	}
}

// RecordCircuitBreakerState records circuit breaker state change (Phase 2)
func (m *LocalAIMetrics) RecordCircuitBreakerState(state string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.circuitBreakerState[state]++
}

// GetMetrics returns current metrics (Phase 2)
func (m *LocalAIMetrics) GetMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	avgLatency := time.Duration(0)
	if m.callCount > 0 {
		avgLatency = m.totalLatency / time.Duration(m.callCount)
	}

	successRate := 0.0
	if m.callCount > 0 {
		successRate = float64(m.successCount) / float64(m.callCount) * 100
	}

	// Calculate average latency per model
	modelAvgLatency := make(map[string]time.Duration)
	for model, count := range m.modelUsage {
		if count > 0 {
			modelAvgLatency[model] = m.modelLatency[model] / time.Duration(count)
		}
	}

	// Calculate average latency per domain
	domainAvgLatency := make(map[string]time.Duration)
	for domain, count := range m.domainUsage {
		if count > 0 {
			domainAvgLatency[domain] = m.domainLatency[domain] / time.Duration(count)
		}
	}

	return map[string]interface{}{
		"call_count":          m.callCount,
		"success_count":       m.successCount,
		"error_count":         m.errorCount,
		"success_rate":        successRate,
		"total_latency":       m.totalLatency.String(),
		"avg_latency":         avgLatency.String(),
		"model_usage":         m.modelUsage,
		"domain_usage":        m.domainUsage,
		"model_avg_latency":   modelAvgLatency,
		"domain_avg_latency":  domainAvgLatency,
		"circuit_breaker_state": m.circuitBreakerState,
		"last_call_time":      m.lastCallTime,
	}
}

// GetMetrics returns metrics for the LocalAI client (Phase 2)
func (c *LocalAIClient) GetMetrics() map[string]interface{} {
	metrics := c.metrics.GetMetrics()
	metrics["circuit_breaker_state"] = c.circuitBreaker.GetState()
	return metrics
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

	oldState := cb.state
	if cb.state == circuitStateHalfOpen {
		cb.state = circuitStateClosed
		cb.failures = 0
	}
	// Note: State change tracking would need access to metrics, handled at client level
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

// GetState returns the current circuit breaker state
func (cb *CircuitBreaker) GetState() string {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// Helper function for power calculation
func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}

