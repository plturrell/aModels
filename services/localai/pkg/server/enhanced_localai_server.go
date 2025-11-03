package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

// EnhancedLocalAIServer provides LocalAI with fallback mechanisms and reliability features
type EnhancedLocalAIServer struct {
	endpoints      []string
	currentIndex   int
	httpClient     *http.Client
	circuitBreaker *CircuitBreaker
	retryConfig    *RetryConfig
	fallbackMode   bool
	mu             sync.RWMutex
}

// CircuitBreaker implements circuit breaker pattern for fault tolerance
type CircuitBreaker struct {
	failureCount     int
	successCount     int
	failureThreshold int
	successThreshold int
	timeout          time.Duration
	lastFailureTime  time.Time
	state            CircuitState
	mu               sync.RWMutex
}

// CircuitState represents the state of the circuit breaker
type CircuitState int

const (
	StateClosed CircuitState = iota
	StateOpen
	StateHalfOpen
)

// FallbackResponse represents a fallback response when LocalAI is unavailable
type FallbackResponse struct {
	Content    string                 `json:"content"`
	Confidence float64                `json:"confidence"`
	Fallback   bool                   `json:"fallback"`
	Reason     string                 `json:"reason"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// NewEnhancedLocalAIServer creates a new enhanced LocalAI server
func NewEnhancedLocalAIServer(endpoints []string) *EnhancedLocalAIServer {
	if len(endpoints) == 0 {
		endpoints = []string{"http://localhost:8080"}
	}

	return &EnhancedLocalAIServer{
		endpoints: endpoints,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		circuitBreaker: &CircuitBreaker{
			failureThreshold: 5,
			successThreshold: 3,
			timeout:          30 * time.Second,
			state:            StateClosed,
		},
		retryConfig: &RetryConfig{
			MaxAttempts: 3,
			BaseDelay:   100 * time.Millisecond,
			MaxDelay:    5 * time.Second,
			Multiplier:  2.0,
			Jitter:      true,
		},
	}
}

// ProcessRequest processes a request with fallback mechanisms
func (s *EnhancedLocalAIServer) ProcessRequest(ctx context.Context, request interface{}) (interface{}, error) {
	// Check circuit breaker state
	if !s.circuitBreaker.CanExecute() {
		return s.handleFallback(ctx, request, "circuit_breaker_open")
	}

	// Try to process with retry logic
	result, err := s.processWithRetry(ctx, request)
	if err != nil {
		// Record failure
		s.circuitBreaker.RecordFailure()

		// Try fallback
		fallbackResult, fallbackErr := s.handleFallback(ctx, request, err.Error())
		if fallbackErr != nil {
			return nil, fmt.Errorf("primary failed: %w, fallback failed: %w", err, fallbackErr)
		}

		return fallbackResult, nil
	}

	// Record success
	s.circuitBreaker.RecordSuccess()
	return result, nil
}

// processWithRetry processes a request with exponential backoff retry
func (s *EnhancedLocalAIServer) processWithRetry(ctx context.Context, request interface{}) (interface{}, error) {
	var lastErr error

	for attempt := 0; attempt <= s.retryConfig.MaxAttempts; attempt++ {
		if attempt > 0 {
			// Calculate delay with exponential backoff
			delay := s.calculateDelay(attempt)

			// Add jitter if enabled
			if s.retryConfig.Jitter {
				delay = s.addJitter(delay)
			}

			// Wait before retry
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		// Try to process request
		result, err := s.processSingleRequest(ctx, request)
		if err == nil {
			return result, nil
		}

		lastErr = err

		// Check if error is retryable
		if !s.isRetryableError(err) {
			break
		}
	}

	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

// processSingleRequest processes a single request to one endpoint
func (s *EnhancedLocalAIServer) processSingleRequest(ctx context.Context, request interface{}) (interface{}, error) {
	s.mu.RLock()
	endpoint := s.endpoints[s.currentIndex]
	s.mu.RUnlock()

	// Try to process with current endpoint
	result, err := s.callEndpoint(ctx, endpoint, request)
	if err != nil {
		// Try next endpoint
		s.mu.Lock()
		s.currentIndex = (s.currentIndex + 1) % len(s.endpoints)
		s.mu.Unlock()

		// Try with next endpoint
		s.mu.RLock()
		nextEndpoint := s.endpoints[s.currentIndex]
		s.mu.RUnlock()

		return s.callEndpoint(ctx, nextEndpoint, request)
	}

	return result, nil
}

// callEndpoint makes a call to a specific endpoint
func (s *EnhancedLocalAIServer) callEndpoint(ctx context.Context, endpoint string, request interface{}) (interface{}, error) {
	// Create HTTP request
	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint+"/v1/chat/completions", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "agenticAiETH-LocalAI-Client/1.0")

	// Make HTTP call
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("endpoint returned status %d", resp.StatusCode)
	}

	// Parse response
	var response map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return response, nil
}

// handleFallback handles fallback when primary processing fails
func (s *EnhancedLocalAIServer) handleFallback(ctx context.Context, request interface{}, reason string) (*FallbackResponse, error) {
	s.mu.Lock()
	s.fallbackMode = true
	s.mu.Unlock()

	// Generate fallback response
	fallback := &FallbackResponse{
		Content:    s.generateFallbackContent(request),
		Confidence: 0.5, // Lower confidence for fallback
		Fallback:   true,
		Reason:     reason,
		Metadata: map[string]interface{}{
			"fallback_time": time.Now().Format(time.RFC3339),
			"reason":        reason,
		},
	}

	return fallback, nil
}

// generateFallbackContent generates basic content when AI is unavailable
func (s *EnhancedLocalAIServer) generateFallbackContent(request interface{}) string {
	// Simple fallback content generation
	// In production, this could use cached responses, rule-based generation, etc.
	return "Fallback response: Unable to process request with AI. Please try again later."
}

// calculateDelay calculates the delay for retry attempts
func (s *EnhancedLocalAIServer) calculateDelay(attempt int) time.Duration {
	delay := float64(s.retryConfig.BaseDelay) * math.Pow(s.retryConfig.Multiplier, float64(attempt-1))

	if delay > float64(s.retryConfig.MaxDelay) {
		delay = float64(s.retryConfig.MaxDelay)
	}

	return time.Duration(delay)
}

// addJitter adds random jitter to delay
func (s *EnhancedLocalAIServer) addJitter(delay time.Duration) time.Duration {
	jitter := time.Duration(rand.Float64() * float64(delay) * 0.1) // 10% jitter
	return delay + jitter
}

// isRetryableError checks if an error is retryable
func (s *EnhancedLocalAIServer) isRetryableError(err error) bool {
	// Check for network errors, timeouts, etc.
	// This is a simplified implementation
	return err != nil
}

// Circuit Breaker Methods

// CanExecute checks if the circuit breaker allows execution
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	switch cb.state {
	case StateClosed:
		return true
	case StateOpen:
		// Check if timeout has passed
		if time.Since(cb.lastFailureTime) > cb.timeout {
			cb.state = StateHalfOpen
			return true
		}
		return false
	case StateHalfOpen:
		return true
	default:
		return false
	}
}

// RecordSuccess records a successful execution
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.successCount++
	cb.failureCount = 0

	if cb.state == StateHalfOpen && cb.successCount >= cb.successThreshold {
		cb.state = StateClosed
		cb.successCount = 0
	}
}

// RecordFailure records a failed execution
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount++
	cb.successCount = 0
	cb.lastFailureTime = time.Now()

	if cb.failureCount >= cb.failureThreshold {
		cb.state = StateOpen
	}
}

// GetState returns the current circuit breaker state
func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// GetStats returns circuit breaker statistics
func (cb *CircuitBreaker) GetStats() map[string]interface{} {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	return map[string]interface{}{
		"state":             cb.state,
		"failure_count":     cb.failureCount,
		"success_count":     cb.successCount,
		"last_failure":      cb.lastFailureTime,
		"failure_threshold": cb.failureThreshold,
		"success_threshold": cb.successThreshold,
	}
}

// HealthCheck performs a health check on all endpoints
func (s *EnhancedLocalAIServer) HealthCheck(ctx context.Context) map[string]interface{} {
	health := make(map[string]interface{})

	for i, endpoint := range s.endpoints {
		start := time.Now()
		_, err := s.callEndpoint(ctx, endpoint, map[string]string{"health_check": "true"})
		duration := time.Since(start)

		health[fmt.Sprintf("endpoint_%d", i)] = map[string]interface{}{
			"url":     endpoint,
			"healthy": err == nil,
			"latency": duration.Milliseconds(),
			"error":   err,
		}
	}

	health["circuit_breaker"] = s.circuitBreaker.GetStats()
	health["fallback_mode"] = s.fallbackMode

	return health
}

// GetMetrics returns server metrics
func (s *EnhancedLocalAIServer) GetMetrics() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return map[string]interface{}{
		"endpoints":        s.endpoints,
		"current_endpoint": s.endpoints[s.currentIndex],
		"fallback_mode":    s.fallbackMode,
		"circuit_breaker":  s.circuitBreaker.GetStats(),
		"retry_config":     s.retryConfig,
	}
}

// UpdateEndpoints updates the list of endpoints
func (s *EnhancedLocalAIServer) UpdateEndpoints(endpoints []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(endpoints) > 0 {
		s.endpoints = endpoints
		s.currentIndex = 0
	}
}

// SetRetryConfig updates the retry configuration
func (s *EnhancedLocalAIServer) SetRetryConfig(config *RetryConfig) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if config != nil {
		s.retryConfig = config
	}
}

// Close closes the enhanced LocalAI server
func (s *EnhancedLocalAIServer) Close() error {
	// Close HTTP client
	s.httpClient.CloseIdleConnections()
	return nil
}
