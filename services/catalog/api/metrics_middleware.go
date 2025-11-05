package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/plturrell/aModels/services/catalog/observability"
)

// MetricsMiddleware provides HTTP middleware for Prometheus metrics.
func MetricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Wrap response writer to capture status code and size
		rw := &responseWriter{
			ResponseWriter: w,
			statusCode:     http.StatusOK,
			size:           0,
		}

		// Process request
		next.ServeHTTP(rw, r)

		// Record metrics
		duration := time.Since(start)
		status := strconv.Itoa(rw.statusCode)
		
		// Estimate request size (simplified)
		requestSize := int64(len(r.Method) + len(r.URL.Path) + len(r.Proto))
		
		observability.RecordRequest(
			r.Method,
			r.URL.Path,
			status,
			duration,
			requestSize,
			int64(rw.size),
		)
	})
}

// responseWriter wraps http.ResponseWriter to capture status and size.
type responseWriter struct {
	http.ResponseWriter
	statusCode int
	size       int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	size, err := rw.ResponseWriter.Write(b)
	rw.size += size
	return size, err
}

