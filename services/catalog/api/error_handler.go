package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// ErrorHandler provides centralized error handling and retry logic.
type ErrorHandler struct {
	logger *log.Logger
}

// NewErrorHandler creates a new error handler.
func NewErrorHandler(logger *log.Logger) *ErrorHandler {
	return &ErrorHandler{
		logger: logger,
	}
}

// APIError represents an API error response.
type APIError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
	RetryAfter *int `json:"retry_after,omitempty"`
	Timestamp string `json:"timestamp"`
}

// HandleError handles an error and writes an appropriate response.
func (eh *ErrorHandler) HandleError(w http.ResponseWriter, r *http.Request, err error, statusCode int) {
	apiErr := APIError{
		Code:      statusCode,
		Message:   http.StatusText(statusCode),
		Details:   err.Error(),
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	// Add retry-after for rate limiting errors
	if statusCode == http.StatusTooManyRequests {
		retryAfter := 60 // seconds
		apiErr.RetryAfter = &retryAfter
	}

	// Log error
	if eh.logger != nil {
		eh.logger.Printf("API Error: %s %s - %d: %v", r.Method, r.URL.Path, statusCode, err)
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": apiErr,
	})
}

// HandlePanic recovers from panics and returns a 500 error.
func (eh *ErrorHandler) HandlePanic(w http.ResponseWriter, r *http.Request) {
	if r := recover(); r != nil {
		var err error
		switch e := r.(type) {
		case error:
			err = e
		case string:
			err = fmt.Errorf("%s", e)
		default:
			err = fmt.Errorf("panic: %v", r)
		}

		eh.HandleError(w, r, err, http.StatusInternalServerError)
	}
}

// RecoveryMiddleware provides panic recovery middleware.
func (eh *ErrorHandler) RecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer eh.HandlePanic(w, r)
		next.ServeHTTP(w, r)
	})
}

// RetryConfig holds retry configuration.
type RetryConfig struct {
	MaxRetries      int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	BackoffMultiplier float64
	RetryableErrors []error
}

// DefaultRetryConfig returns default retry configuration.
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:       3,
		InitialDelay:     100 * time.Millisecond,
		MaxDelay:         5 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// IsRetryableError checks if an error is retryable.
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}

	// Network errors are retryable
	errStr := err.Error()
	if contains(errStr, "timeout") || contains(errStr, "connection") || contains(errStr, "network") {
		return true
	}

	// 5xx errors are retryable
	if contains(errStr, "500") || contains(errStr, "502") || contains(errStr, "503") || contains(errStr, "504") {
		return true
	}

	// 429 (rate limit) is retryable
	if contains(errStr, "429") || contains(errStr, "rate limit") {
		return true
	}

	return false
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || 
		(len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || 
		findSubstring(s, substr))))
}

func findSubstring(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

