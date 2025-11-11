package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

// Logger interface for health handler logging
type Logger interface {
	Info(msg string, fields map[string]interface{})
	Error(msg string, err error, fields map[string]interface{})
}

// HealthHandler provides enhanced health check endpoints.
type HealthHandler struct {
	checkers []HealthChecker
	logger   Logger
}

// HealthChecker defines a health check.
type HealthChecker interface {
	Check(ctx context.Context) HealthStatus
	Name() string
}

// HealthStatus represents the status of a health check.
type HealthStatus struct {
	Status    string            `json:"status"` // "ok", "degraded", "down"
	Message   string            `json:"message,omitempty"`
	Details   map[string]any    `json:"details,omitempty"`
	Timestamp time.Time         `json:"timestamp"`
}

// NewHealthHandler creates a new health handler.
func NewHealthHandler(
	checkers []HealthChecker,
	logger Logger,
) *HealthHandler {
	return &HealthHandler{
		checkers: checkers,
		logger:   logger,
	}
}

// HandleHealthz handles GET /healthz.
func (h *HealthHandler) HandleHealthz(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	statuses := make(map[string]HealthStatus)
	overallStatus := "ok"

	for _, checker := range h.checkers {
		status := checker.Check(ctx)
		statuses[checker.Name()] = status

		if status.Status == "down" {
			overallStatus = "down"
		} else if status.Status == "degraded" && overallStatus == "ok" {
			overallStatus = "degraded"
		}
	}

	response := map[string]any{
		"status":   overallStatus,
		"service":  "catalog",
		"timestamp": time.Now(),
		"checks":   statuses,
	}

	statusCode := http.StatusOK
	if overallStatus == "down" {
		statusCode = http.StatusServiceUnavailable
	} else if overallStatus == "degraded" {
		statusCode = http.StatusOK // Still OK, but degraded
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}

// HandleReadiness handles GET /ready.
func (h *HealthHandler) HandleReadiness(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	// Check critical dependencies
	ready := true
	for _, checker := range h.checkers {
		status := checker.Check(ctx)
		if status.Status == "down" {
			ready = false
			break
		}
	}

	if !ready {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("not ready"))
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ready"))
}

// HandleLiveness handles GET /live.
func (h *HealthHandler) HandleLiveness(w http.ResponseWriter, r *http.Request) {
	// Liveness check is simple - service is alive if it can respond
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("alive"))
}

// BasicHealthChecker provides a basic health check.
type BasicHealthChecker struct {
	name     string
	checkFn  func(ctx context.Context) HealthStatus
}

// NewBasicHealthChecker creates a basic health checker.
func NewBasicHealthChecker(name string, checkFn func(ctx context.Context) HealthStatus) *BasicHealthChecker {
	return &BasicHealthChecker{
		name:    name,
		checkFn: checkFn,
	}
}

// Name returns the checker name.
func (b *BasicHealthChecker) Name() string {
	return b.name
}

// Check performs the health check.
func (b *BasicHealthChecker) Check(ctx context.Context) HealthStatus {
	return b.checkFn(ctx)
}

