package api

import (
	"context"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/catalog/observability"
)

// MonitoringMiddleware provides monitoring and alerting capabilities.
type MonitoringMiddleware struct {
	logger *log.Logger
}

// NewMonitoringMiddleware creates a new monitoring middleware.
func NewMonitoringMiddleware(logger *log.Logger) *MonitoringMiddleware {
	return &MonitoringMiddleware{
		logger: logger,
	}
}

// AlertConfig holds alert configuration.
type AlertConfig struct {
	MaxLatency     time.Duration
	ErrorThreshold float64 // Percentage
	CheckInterval  time.Duration
}

// DefaultAlertConfig returns default alert configuration.
func DefaultAlertConfig() *AlertConfig {
	return &AlertConfig{
		MaxLatency:     2 * time.Second,
		ErrorThreshold: 5.0, // 5%
		CheckInterval:  30 * time.Second,
	}
}

// MonitorRequest monitors a request and checks for alerts.
func (mm *MonitoringMiddleware) MonitorRequest(ctx context.Context, method, path string, duration time.Duration, statusCode int) {
	// Record metrics
	statusStr := "200"
	if statusCode >= 200 && statusCode < 300 {
		statusStr = "2xx"
	} else if statusCode >= 300 && statusCode < 400 {
		statusStr = "3xx"
	} else if statusCode >= 400 && statusCode < 500 {
		statusStr = "4xx"
	} else if statusCode >= 500 {
		statusStr = "5xx"
	}
	observability.RecordRequest(method, path, statusStr, duration, 0, 0)

	// Check for alerts
	if duration > DefaultAlertConfig().MaxLatency {
		mm.alert("high_latency", map[string]interface{}{
			"method":     method,
			"path":       path,
			"duration_ms": duration.Milliseconds(),
			"threshold_ms": DefaultAlertConfig().MaxLatency.Milliseconds(),
		})
	}

	if statusCode >= 500 {
		mm.alert("server_error", map[string]interface{}{
			"method":     method,
			"path":       path,
			"status_code": statusCode,
		})
	}
}

// alert sends an alert.
func (mm *MonitoringMiddleware) alert(alertType string, details map[string]interface{}) {
	if mm.logger != nil {
		mm.logger.Printf("ALERT [%s]: %v", alertType, details)
	}

	// In production, would send to alerting system (PagerDuty, Slack, etc.)
	observability.RecordAlert(alertType, details)
}

// Middleware wraps a handler with monitoring.
func (mm *MonitoringMiddleware) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap response writer to capture status
		rw := &responseWriter{
			ResponseWriter: w,
			statusCode:     http.StatusOK,
		}

		// Process request
		next.ServeHTTP(rw, r)

		// Monitor
		duration := time.Since(start)
		mm.MonitorRequest(r.Context(), r.Method, r.URL.Path, duration, rw.statusCode)
	})
}

