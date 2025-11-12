package middleware

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

// HealthChecker provides health check functionality
type HealthChecker struct {
	checks      []HealthCheck
	logger      *log.Logger
	lastCheck   time.Time
	lastStatus  HealthStatus
	mu          sync.RWMutex
	checkInterval time.Duration
}

// HealthCheck represents a single health check
type HealthCheck struct {
	Name    string
	Check   func(ctx context.Context) error
	Timeout time.Duration
}

// HealthStatus represents the overall health status
type HealthStatus struct {
	Status    string                 `json:"status"` // "healthy", "degraded", "unhealthy"
	Timestamp time.Time              `json:"timestamp"`
	Checks    map[string]CheckResult `json:"checks"`
}

// CheckResult represents the result of a single check
type CheckResult struct {
	Status    string        `json:"status"` // "pass", "fail", "warn"
	Message   string        `json:"message,omitempty"`
	Duration  time.Duration `json:"duration_ms"`
	Timestamp time.Time     `json:"timestamp"`
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(logger *log.Logger) *HealthChecker {
	return &HealthChecker{
		checks:        make([]HealthCheck, 0),
		logger:         logger,
		checkInterval:  30 * time.Second, // Cache results for 30 seconds
		lastStatus: HealthStatus{
			Status:    "unknown",
			Timestamp: time.Now(),
			Checks:    make(map[string]CheckResult),
		},
	}
}

// RegisterCheck registers a health check
func (hc *HealthChecker) RegisterCheck(name string, check func(ctx context.Context) error, timeout time.Duration) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	hc.checks = append(hc.checks, HealthCheck{
		Name:    name,
		Check:   check,
		Timeout: timeout,
	})
}

// Check performs all registered health checks
func (hc *HealthChecker) Check(ctx context.Context) HealthStatus {
	hc.mu.RLock()
	// Use cached result if recent
	if time.Since(hc.lastCheck) < hc.checkInterval && hc.lastStatus.Status != "unknown" {
		hc.mu.RUnlock()
		return hc.lastStatus
	}
	hc.mu.RUnlock()

	hc.mu.Lock()
	defer hc.mu.Unlock()

	// Double-check after acquiring write lock
	if time.Since(hc.lastCheck) < hc.checkInterval && hc.lastStatus.Status != "unknown" {
		return hc.lastStatus
	}

	status := HealthStatus{
		Status:    "healthy",
		Timestamp: time.Now(),
		Checks:    make(map[string]CheckResult),
	}

	// Run all checks
	for _, check := range hc.checks {
		checkCtx, cancel := context.WithTimeout(ctx, check.Timeout)
		start := time.Now()
		
		err := check.Check(checkCtx)
		duration := time.Since(start)
		cancel()

		result := CheckResult{
			Status:    "pass",
			Duration:  duration,
			Timestamp: time.Now(),
		}

		if err != nil {
			result.Status = "fail"
			result.Message = err.Error()
			// If any critical check fails, mark as unhealthy
			status.Status = "unhealthy"
		}

		status.Checks[check.Name] = result
	}

	// If no checks registered, mark as healthy
	if len(hc.checks) == 0 {
		status.Status = "healthy"
	}

	hc.lastCheck = time.Now()
	hc.lastStatus = status

	return status
}

// HandleHealth handles /health endpoint
func (hc *HealthChecker) HandleHealth(w http.ResponseWriter, r *http.Request) {
	status := hc.Check(r.Context())
	
	statusCode := http.StatusOK
	if status.Status == "unhealthy" {
		statusCode = http.StatusServiceUnavailable
	} else if status.Status == "degraded" {
		statusCode = http.StatusOK // Still 200, but indicates degraded state
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(status)
}

// HandleReady handles /ready endpoint (readiness probe)
func (hc *HealthChecker) HandleReady(w http.ResponseWriter, r *http.Request) {
	status := hc.Check(r.Context())
	
	// Readiness requires all checks to pass
	ready := status.Status == "healthy"
	
	statusCode := http.StatusOK
	if !ready {
		statusCode = http.StatusServiceUnavailable
	}

	response := map[string]interface{}{
		"ready":     ready,
		"status":    status.Status,
		"timestamp": status.Timestamp,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}

// DatabaseHealthCheck creates a health check for a database connection
func DatabaseHealthCheck(db *sql.DB, name string) HealthCheck {
	return HealthCheck{
		Name:    name,
		Timeout: 5 * time.Second,
		Check: func(ctx context.Context) error {
			if db == nil {
				return fmt.Errorf("database connection is nil")
			}
			return db.PingContext(ctx)
		},
	}
}

// HTTPHealthCheck creates a health check for an HTTP endpoint
func HTTPHealthCheck(client *http.Client, url, name string) HealthCheck {
	return HealthCheck{
		Name:    name,
		Timeout: 5 * time.Second,
		Check: func(ctx context.Context) error {
			req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
			if err != nil {
				return fmt.Errorf("failed to create request: %w", err)
			}
			resp, err := client.Do(req)
			if err != nil {
				return fmt.Errorf("request failed: %w", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode >= 400 {
				return fmt.Errorf("endpoint returned status %d", resp.StatusCode)
			}
			return nil
		},
	}
}

