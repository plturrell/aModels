package rest

import (
	"log"
	"net/http"
	"time"
)

// MetricsCollector collects metrics for observability
type MetricsCollector struct {
	requestCount    int64
	errorCount      int64
	totalLatency    time.Duration
	requestLatencies []time.Duration
	logger          *log.Logger
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(logger *log.Logger) *MetricsCollector {
	return &MetricsCollector{
		requestLatencies: make([]time.Duration, 0, 1000),
		logger:           logger,
	}
}

// RecordRequest records a request with its latency
func (m *MetricsCollector) RecordRequest(latency time.Duration) {
	m.requestCount++
	m.totalLatency += latency
	if len(m.requestLatencies) < 1000 {
		m.requestLatencies = append(m.requestLatencies, latency)
	}
}

// RecordError records an error
func (m *MetricsCollector) RecordError() {
	m.errorCount++
}

// GetMetrics returns current metrics
func (m *MetricsCollector) GetMetrics() map[string]interface{} {
	avgLatency := time.Duration(0)
	if m.requestCount > 0 {
		avgLatency = m.totalLatency / time.Duration(m.requestCount)
	}
	
	errorRate := 0.0
	if m.requestCount > 0 {
		errorRate = float64(m.errorCount) / float64(m.requestCount)
	}
	
	return map[string]interface{}{
		"request_count":   m.requestCount,
		"error_count":     m.errorCount,
		"average_latency": avgLatency.String(),
		"error_rate":      errorRate,
	}
}

// ObservabilityMiddleware adds observability to HTTP handlers
func ObservabilityMiddleware(metrics *MetricsCollector, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Wrap response writer to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		
		next.ServeHTTP(wrapped, r)
		
		latency := time.Since(start)
		metrics.RecordRequest(latency)
		
		if wrapped.statusCode >= 400 {
			metrics.RecordError()
		}
	})
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// LoggingMiddleware adds structured logging
func LoggingMiddleware(logger *log.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(wrapped, r)
		
		latency := time.Since(start)
		
		logger.Printf(
			"method=%s path=%s status=%d latency=%s remote_addr=%s",
			r.Method,
			r.URL.Path,
			wrapped.statusCode,
			latency,
			r.RemoteAddr,
		)
	})
}

