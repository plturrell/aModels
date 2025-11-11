// Package integration provides standardized error handling and logging utilities
// for cross-service integration in the lang infrastructure.
//
// These utilities provide:
// - Consistent retry logic with exponential backoff
// - Structured logging with correlation IDs
// - HTTP request/response logging
// - Integration call tracking
package integration

import (
	"context"
	"fmt"
	"log"
	"time"
)

// RetryConfig holds configuration for retry logic.
// Used to customize retry behavior for different operations.
type RetryConfig struct {
	MaxRetries      int           // Maximum number of retry attempts (default: 3)
	InitialDelay    time.Duration // Initial delay before first retry (default: 1s)
	MaxDelay        time.Duration // Maximum delay between retries (default: 30s)
	BackoffMultiplier float64     // Multiplier for exponential backoff (default: 2.0)
}

// DefaultRetryConfig returns a default retry configuration.
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:       3,
		InitialDelay:     1 * time.Second,
		MaxDelay:         30 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// IsRetryableError determines if an error should be retried.
// Returns true for network errors, 5xx HTTP errors, and rate limiting (429).
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()
	
	// Network errors are retryable
	if contains(errStr, "timeout") || contains(errStr, "connection") || 
		contains(errStr, "network") || contains(errStr, "dial") {
		return true
	}

	// 5xx server errors are retryable
	if contains(errStr, "500") || contains(errStr, "502") || 
		contains(errStr, "503") || contains(errStr, "504") {
		return true
	}

	// 429 (rate limit) is retryable
	if contains(errStr, "429") || contains(errStr, "rate limit") {
		return true
	}

	return false
}

// IsRetryableHTTPStatus determines if an HTTP status code should be retried.
func IsRetryableHTTPStatus(statusCode int) bool {
	// 5xx server errors
	if statusCode >= 500 && statusCode < 600 {
		return true
	}
	// 429 rate limiting
	if statusCode == 429 {
		return true
	}
	return false
}

// RetryWithBackoff executes a function with exponential backoff retry logic.
// The function will be retried if it returns a retryable error.
func RetryWithBackoff(
	ctx context.Context,
	config *RetryConfig,
	logger *log.Logger,
	operationName string,
	operation func() error,
) error {
	if config == nil {
		config = DefaultRetryConfig()
	}

	var lastErr error
	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		if attempt > 0 {
			// Calculate exponential backoff delay
			delay := time.Duration(float64(config.InitialDelay) * 
				pow(config.BackoffMultiplier, float64(attempt-1)))
			if delay > config.MaxDelay {
				delay = config.MaxDelay
			}

			if logger != nil {
				logger.Printf("[RETRY] %s: attempt %d/%d after %v", 
					operationName, attempt+1, config.MaxRetries+1, delay)
			}

			select {
			case <-ctx.Done():
				return fmt.Errorf("%s: context cancelled: %w", operationName, ctx.Err())
			case <-time.After(delay):
			}
		}

		err := operation()
		if err == nil {
			if attempt > 0 && logger != nil {
				logger.Printf("[SUCCESS] %s: succeeded on attempt %d", 
					operationName, attempt+1)
			}
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !IsRetryableError(err) {
			if logger != nil {
				logger.Printf("[ERROR] %s: non-retryable error: %v", operationName, err)
			}
			return fmt.Errorf("%s: non-retryable error: %w", operationName, err)
		}

		if logger != nil {
			logger.Printf("[RETRY] %s: attempt %d failed: %v", 
				operationName, attempt+1, err)
		}
	}

	if logger != nil {
		logger.Printf("[ERROR] %s: failed after %d retries: %v", 
			operationName, config.MaxRetries+1, lastErr)
	}
	return fmt.Errorf("%s: failed after %d retries: %w", 
		operationName, config.MaxRetries+1, lastErr)
}

// RetryWithBackoffResult executes a function with exponential backoff retry logic
// that returns a result.
// The function will be retried if it returns a retryable error.
func RetryWithBackoffResult[T any](
	ctx context.Context,
	config *RetryConfig,
	logger *log.Logger,
	operationName string,
	operation func() (T, error),
) (T, error) {
	var zero T
	if config == nil {
		config = DefaultRetryConfig()
	}

	var lastErr error
	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		if attempt > 0 {
			// Calculate exponential backoff delay
			delay := time.Duration(float64(config.InitialDelay) * 
				pow(config.BackoffMultiplier, float64(attempt-1)))
			if delay > config.MaxDelay {
				delay = config.MaxDelay
			}

			if logger != nil {
				logger.Printf("[RETRY] %s: attempt %d/%d after %v", 
					operationName, attempt+1, config.MaxRetries+1, delay)
			}

			select {
			case <-ctx.Done():
				return zero, fmt.Errorf("%s: context cancelled: %w", operationName, ctx.Err())
			case <-time.After(delay):
			}
		}

		result, err := operation()
		if err == nil {
			if attempt > 0 && logger != nil {
				logger.Printf("[SUCCESS] %s: succeeded on attempt %d", 
					operationName, attempt+1)
			}
			return result, nil
		}

		lastErr = err

		// Check if error is retryable
		if !IsRetryableError(err) {
			if logger != nil {
				logger.Printf("[ERROR] %s: non-retryable error: %v", operationName, err)
			}
			return zero, fmt.Errorf("%s: non-retryable error: %w", operationName, err)
		}

		if logger != nil {
			logger.Printf("[RETRY] %s: attempt %d failed: %v", 
				operationName, attempt+1, err)
		}
	}

	if logger != nil {
		logger.Printf("[ERROR] %s: failed after %d retries: %v", 
			operationName, config.MaxRetries+1, lastErr)
	}
	return zero, fmt.Errorf("%s: failed after %d retries: %w", 
		operationName, config.MaxRetries+1, lastErr)
}

// Helper functions

func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		(s == substr || 
			(len(s) > len(substr) && 
				(s[:len(substr)] == substr || 
					s[len(s)-len(substr):] == substr ||
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

func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}

