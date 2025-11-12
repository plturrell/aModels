package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

// SelfHealingSystem provides automatic error detection and recovery.
// Phase 9.2: Enhanced with domain health monitoring for domain-aware self-healing.
type SelfHealingSystem struct {
	logger            *log.Logger
	retryConfig       *RetryConfig
	circuitBreakers   map[string]*CircuitBreaker
	fallbackHandlers  map[string]FallbackHandler
	healthMonitors    map[string]*HealthMonitor
	domainDetector    *DomainDetector         // Phase 9.2: Domain detector for domain health
	domainHealthCache map[string]DomainHealth // Phase 9.2: domain_id -> health score
	mu                sync.RWMutex
}

// DomainHealth represents domain health status.
type DomainHealth struct {
	DomainID  string
	Score     float64 // 0.0 to 1.0
	Status    string  // "healthy", "degraded", "unhealthy"
	LastCheck time.Time
	Metrics   map[string]any
}

// RetryConfig holds retry configuration.
type RetryConfig struct {
	MaxRetries        int
	InitialDelay      time.Duration
	MaxDelay          time.Duration
	BackoffMultiplier float64
}

// DefaultRetryConfig returns default retry configuration.
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:        3,
		InitialDelay:      2 * time.Second,
		MaxDelay:          60 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// CircuitBreaker implements the circuit breaker pattern.
type CircuitBreaker struct {
	name            string
	failureCount    int
	successCount    int
	lastFailureTime time.Time
	state           CircuitState
	threshold       int
	timeout         time.Duration
	mu              sync.RWMutex
}

// CircuitState represents the state of a circuit breaker.
type CircuitState string

const (
	CircuitStateClosed   CircuitState = "closed"    // Normal operation
	CircuitStateOpen     CircuitState = "open"      // Failing, reject requests
	CircuitStateHalfOpen CircuitState = "half_open" // Testing if recovered
)

// FallbackHandler handles fallback operations.
type FallbackHandler func(ctx context.Context, err error) (any, error)

// HealthMonitor monitors the health of a service.
type HealthMonitor struct {
	serviceName   string
	lastCheck     time.Time
	lastStatus    bool
	failureCount  int
	successCount  int
	checkInterval time.Duration
	healthCheck   func() error
	mu            sync.RWMutex
}

// NewSelfHealingSystem creates a new self-healing system.
func NewSelfHealingSystem(logger *log.Logger) *SelfHealingSystem {
	localaiURL := os.Getenv("LOCALAI_URL")
	var domainDetector *DomainDetector
	if localaiURL != "" {
		domainDetector = NewDomainDetector(localaiURL, logger)
	}

	return &SelfHealingSystem{
		logger:            logger,
		retryConfig:       DefaultRetryConfig(),
		circuitBreakers:   make(map[string]*CircuitBreaker),
		fallbackHandlers:  make(map[string]FallbackHandler),
		healthMonitors:    make(map[string]*HealthMonitor),
		domainDetector:    domainDetector,                // Phase 9.2: Domain detector
		domainHealthCache: make(map[string]DomainHealth), // Phase 9.2: Domain health cache
	}
}

// GetDomainHealth gets the health score for a domain.
// Phase 9.2: Domain health monitoring using metrics collector.
func (shs *SelfHealingSystem) GetDomainHealth(domainID string) DomainHealth {
	// Check cache first
	shs.mu.RLock()
	if cached, exists := shs.domainHealthCache[domainID]; exists {
		// Return cached if less than 5 minutes old
		if time.Since(cached.LastCheck) < 5*time.Minute {
			shs.mu.RUnlock()
			return cached
		}
	}
	shs.mu.RUnlock()

	// Fetch fresh health data
	health := shs.fetchDomainHealth(domainID)

	// Update cache
	shs.mu.Lock()
	shs.domainHealthCache[domainID] = health
	shs.mu.Unlock()

	return health
}

// fetchDomainHealth fetches domain health from PostgreSQL/metrics collector.
func (shs *SelfHealingSystem) fetchDomainHealth(domainID string) DomainHealth {
	health := DomainHealth{
		DomainID:  domainID,
		Score:     0.5, // Default
		Status:    "unknown",
		LastCheck: time.Now(),
		Metrics:   make(map[string]any),
	}

	// Try to get domain metrics from PostgreSQL
	postgresDSN := os.Getenv("POSTGRES_DSN")
	if postgresDSN != "" {
		// Query PostgreSQL for domain metrics
		// This would call DomainMetricsCollector in Python or query directly
		// For now, use a simple HTTP call to training service if available
		trainingURL := os.Getenv("TRAINING_SERVICE_URL")
		if trainingURL != "" {
			metricsURL := fmt.Sprintf("%s/domain-metrics/%s", trainingURL, domainID)
			resp, err := http.Get(metricsURL)
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				body, _ := io.ReadAll(resp.Body)
				var metricsData map[string]any
				if json.Unmarshal(body, &metricsData) == nil {
					// Extract health score from metrics
					if performance, ok := metricsData["performance"].(map[string]any); ok {
						if latest, ok := performance["latest"].(map[string]any); ok {
							// Calculate health score from metrics
							accuracy := 0.0
							latency := 0.0

							if acc, ok := latest["accuracy"].(float64); ok {
								accuracy = acc
							}
							if lat, ok := latest["latency_ms"].(float64); ok {
								latency = lat
							}

							// Health score: accuracy weighted, latency penalty
							healthScore := accuracy * 0.8
							if latency > 1000 {
								healthScore *= 0.9 // Penalty for high latency
							}
							if latency > 2000 {
								healthScore *= 0.8
							}

							health.Score = healthScore
							health.Metrics = latest

							// Determine status
							if healthScore >= 0.8 {
								health.Status = "healthy"
							} else if healthScore >= 0.5 {
								health.Status = "degraded"
							} else {
								health.Status = "unhealthy"
							}
						}
					}
				}
			}
		}
	}

	return health
}

// ExecuteWithRetry executes a function with automatic retry.
func (shs *SelfHealingSystem) ExecuteWithRetry(
	ctx context.Context,
	operationName string,
	operation func() error,
) error {
	var lastErr error

	for attempt := 0; attempt <= shs.retryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			// Calculate backoff delay
			delay := time.Duration(float64(shs.retryConfig.InitialDelay) *
				pow(shs.retryConfig.BackoffMultiplier, float64(attempt-1)))
			if delay > shs.retryConfig.MaxDelay {
				delay = shs.retryConfig.MaxDelay
			}

			shs.logger.Printf("Retrying %s (attempt %d) after %v", operationName, attempt, delay)

			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}

		// Execute operation
		err := operation()
		if err == nil {
			shs.logger.Printf("Operation %s succeeded on attempt %d", operationName, attempt+1)
			return nil
		}

		lastErr = err
		shs.logger.Printf("Operation %s failed (attempt %d): %v", operationName, attempt+1, err)
	}

	shs.logger.Printf("Operation %s failed after %d retries", operationName, shs.retryConfig.MaxRetries)
	return fmt.Errorf("operation %s failed after %d retries: %w", operationName, shs.retryConfig.MaxRetries, lastErr)
}

// ExecuteWithCircuitBreaker executes a function with circuit breaker protection.
// Phase 9.2: Enhanced with domain health monitoring.
func (shs *SelfHealingSystem) ExecuteWithCircuitBreaker(
	ctx context.Context,
	serviceName string,
	operation func() (any, error),
	domainID string, // Phase 9.2: Optional domain ID for domain health check
) (any, error) {
	// Phase 9.2: Check domain health if domainID provided
	if domainID != "" && shs.domainDetector != nil {
		domainHealth := shs.GetDomainHealth(domainID)

		if domainHealth.Score < 0.5 {
			// Domain unhealthy, use fallback immediately
			fallbackKey := fmt.Sprintf("%s_%s", serviceName, domainID)
			if fallback, exists := shs.fallbackHandlers[fallbackKey]; exists {
				shs.logger.Printf(
					"Domain %s health low (%.2f), using fallback for %s",
					domainID, domainHealth.Score, serviceName,
				)
				return fallback(ctx, fmt.Errorf("domain_unhealthy: score=%.2f", domainHealth.Score))
			}
			// Try generic fallback
			if fallback, exists := shs.fallbackHandlers[serviceName]; exists {
				shs.logger.Printf(
					"Domain %s health low (%.2f), using generic fallback for %s",
					domainID, domainHealth.Score, serviceName,
				)
				return fallback(ctx, fmt.Errorf("domain_unhealthy: score=%.2f", domainHealth.Score))
			}
		}
	}

	// Get or create circuit breaker
	cb := shs.getOrCreateCircuitBreaker(serviceName)

	// Check circuit state
	cb.mu.RLock()
	state := cb.state
	cb.mu.RUnlock()

	if state == CircuitStateOpen {
		// Circuit is open, check if we should try half-open
		cb.mu.Lock()
		if time.Since(cb.lastFailureTime) > cb.timeout {
			cb.state = CircuitStateHalfOpen
			cb.mu.Unlock()
			shs.logger.Printf("Circuit breaker %s: transitioning to half-open", serviceName)
		} else {
			cb.mu.Unlock()
			// Check for fallback
			fallbackKey := fmt.Sprintf("%s_%s", serviceName, domainID)
			if domainID != "" {
				if fallback, exists := shs.fallbackHandlers[fallbackKey]; exists {
					shs.logger.Printf("Circuit breaker %s: using domain-specific fallback for %s", serviceName, domainID)
					return fallback(ctx, fmt.Errorf("circuit breaker is open"))
				}
			}
			if fallback, exists := shs.fallbackHandlers[serviceName]; exists {
				shs.logger.Printf("Circuit breaker %s: using fallback handler", serviceName)
				return fallback(ctx, fmt.Errorf("circuit breaker is open"))
			}
			return nil, fmt.Errorf("circuit breaker %s is open", serviceName)
		}
	}

	// Execute operation
	result, err := operation()

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		// Operation failed
		cb.failureCount++
		cb.successCount = 0
		cb.lastFailureTime = time.Now()

		if cb.failureCount >= cb.threshold {
			cb.state = CircuitStateOpen
			shs.logger.Printf("Circuit breaker %s: opened (failure count: %d)", serviceName, cb.failureCount)
		}

		return result, err
	}

	// Operation succeeded
	cb.successCount++
	if cb.state == CircuitStateHalfOpen {
		// Successful operation in half-open state, close the circuit
		cb.state = CircuitStateClosed
		cb.failureCount = 0
		shs.logger.Printf("Circuit breaker %s: closed (recovered)", serviceName)
	}

	return result, nil
}

// RegisterFallbackHandler registers a fallback handler for a service.
func (shs *SelfHealingSystem) RegisterFallbackHandler(
	serviceName string,
	handler FallbackHandler,
) {
	shs.mu.Lock()
	defer shs.mu.Unlock()
	shs.fallbackHandlers[serviceName] = handler
	shs.logger.Printf("Registered fallback handler for %s", serviceName)
}

// RegisterHealthMonitor registers a health monitor for a service.
func (shs *SelfHealingSystem) RegisterHealthMonitor(
	serviceName string,
	checkInterval time.Duration,
	healthCheck func() error,
) {
	shs.mu.Lock()
	defer shs.mu.Unlock()

	monitor := &HealthMonitor{
		serviceName:   serviceName,
		checkInterval: checkInterval,
		healthCheck:   healthCheck,
		lastStatus:    true,
	}

	shs.healthMonitors[serviceName] = monitor

	// Start monitoring goroutine
	go shs.monitorHealth(monitor)

	shs.logger.Printf("Registered health monitor for %s (interval: %v)", serviceName, checkInterval)
}

// monitorHealth continuously monitors service health.
func (shs *SelfHealingSystem) monitorHealth(monitor *HealthMonitor) {
	ticker := time.NewTicker(monitor.checkInterval)
	defer ticker.Stop()

	for range ticker.C {
		err := monitor.healthCheck()

		monitor.mu.Lock()
		monitor.lastCheck = time.Now()

		if err != nil {
			monitor.lastStatus = false
			monitor.failureCount++
			monitor.successCount = 0
			shs.logger.Printf("Health check failed for %s: %v", monitor.serviceName, err)
		} else {
			monitor.lastStatus = true
			monitor.successCount++
			if monitor.failureCount > 0 {
				monitor.failureCount = 0
			}
		}
		monitor.mu.Unlock()
	}
}

// GetHealthStatus returns the health status of a service.
func (shs *SelfHealingSystem) GetHealthStatus(serviceName string) (bool, error) {
	shs.mu.RLock()
	monitor, exists := shs.healthMonitors[serviceName]
	shs.mu.RUnlock()

	if !exists {
		return false, fmt.Errorf("health monitor not found for %s", serviceName)
	}

	monitor.mu.RLock()
	defer monitor.mu.RUnlock()

	return monitor.lastStatus, nil
}

// getOrCreateCircuitBreaker gets or creates a circuit breaker for a service.
func (shs *SelfHealingSystem) getOrCreateCircuitBreaker(serviceName string) *CircuitBreaker {
	shs.mu.Lock()
	defer shs.mu.Unlock()

	if cb, exists := shs.circuitBreakers[serviceName]; exists {
		return cb
	}

	cb := &CircuitBreaker{
		name:      serviceName,
		state:     CircuitStateClosed,
		threshold: 5,                // Open after 5 failures
		timeout:   30 * time.Second, // Try half-open after 30 seconds
	}

	shs.circuitBreakers[serviceName] = cb
	return cb
}

// Helper function
func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}
