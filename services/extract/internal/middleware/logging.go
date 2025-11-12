package middleware

import (
	"encoding/json"
	"log"
	"net/http"
	"time"
)

// StructuredLogger provides structured JSON logging
type StructuredLogger struct {
	logger *log.Logger
}

// LogEntry represents a structured log entry
type LogEntry struct {
	Timestamp time.Time              `json:"timestamp"`
	Level     string                 `json:"level"`
	Message   string                 `json:"message"`
	Service   string                 `json:"service"`
	Fields    map[string]interface{} `json:"fields,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// NewStructuredLogger creates a new structured logger
func NewStructuredLogger(logger *log.Logger) *StructuredLogger {
	return &StructuredLogger{
		logger: logger,
	}
}

// Log writes a structured log entry
func (sl *StructuredLogger) Log(level, message string, fields map[string]interface{}, err error) {
	entry := LogEntry{
		Timestamp: time.Now(),
		Level:     level,
		Message:   message,
		Service:   "extract",
		Fields:    fields,
	}

	if err != nil {
		entry.Error = err.Error()
	}

	jsonData, jsonErr := json.Marshal(entry)
	if jsonErr != nil {
		// Fallback to simple logging if JSON marshaling fails
		sl.logger.Printf("[%s] %s: %v", level, message, fields)
		if err != nil {
			sl.logger.Printf("Error: %v", err)
		}
		return
	}

	sl.logger.Println(string(jsonData))
}

// Debug logs a debug message
func (sl *StructuredLogger) Debug(message string, fields map[string]interface{}) {
	sl.Log("DEBUG", message, fields, nil)
}

// Info logs an info message
func (sl *StructuredLogger) Info(message string, fields map[string]interface{}) {
	sl.Log("INFO", message, fields, nil)
}

// Warn logs a warning message
func (sl *StructuredLogger) Warn(message string, fields map[string]interface{}) {
	sl.Log("WARN", message, fields, nil)
}

// Error logs an error message
func (sl *StructuredLogger) Error(message string, err error, fields map[string]interface{}) {
	if fields == nil {
		fields = make(map[string]interface{})
	}
	sl.Log("ERROR", message, fields, err)
}

// LoggingMiddleware provides HTTP request/response logging
func LoggingMiddleware(logger *StructuredLogger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()

			// Create a response writer wrapper to capture status code
			rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

			// Process request
			next.ServeHTTP(rw, r)

			// Log request
			duration := time.Since(start)
			logger.Info("http_request", map[string]interface{}{
				"method":      r.Method,
				"path":        r.URL.Path,
				"status_code": rw.statusCode,
				"duration_ms": duration.Milliseconds(),
				"remote_addr": r.RemoteAddr,
				"user_agent":  r.UserAgent(),
			})
		})
	}
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

