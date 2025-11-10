package retry

import (
	"context"
	"math/rand"
	"time"
)

// Config configures retry behavior.
type Config struct {
	MaxAttempts   int           // Maximum number of attempts (default: 3)
	InitialBackoff time.Duration // Initial backoff duration (default: 100ms)
	MaxBackoff     time.Duration // Maximum backoff duration (default: 5s)
	Jitter         bool         // Add random jitter to backoff (default: true)
	Retryable      func(error) bool // Function to determine if error is retryable (default: all errors)
}

// DefaultConfig returns default retry configuration.
func DefaultConfig() Config {
	return Config{
		MaxAttempts:   3,
		InitialBackoff: 100 * time.Millisecond,
		MaxBackoff:     5 * time.Second,
		Jitter:         true,
		Retryable: func(err error) bool {
			return err != nil // Retry all errors by default
		},
	}
}

// WithRetry executes a function with retry logic using exponential backoff.
func WithRetry(ctx context.Context, config Config, fn func() error) error {
	if config.MaxAttempts <= 0 {
		config.MaxAttempts = 3
	}
	if config.InitialBackoff <= 0 {
		config.InitialBackoff = 100 * time.Millisecond
	}
	if config.MaxBackoff <= 0 {
		config.MaxBackoff = 5 * time.Second
	}
	if config.Retryable == nil {
		config.Retryable = func(err error) bool {
			return err != nil
		}
	}

	var lastErr error
	for attempt := 0; attempt < config.MaxAttempts; attempt++ {
		if attempt > 0 {
			backoff := calculateBackoff(attempt, config.InitialBackoff, config.MaxBackoff, config.Jitter)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err
		if !config.Retryable(err) {
			return err
		}

		// Check if context is cancelled before retrying
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	return lastErr
}

// calculateBackoff calculates exponential backoff with optional jitter.
func calculateBackoff(attempt int, initial, max time.Duration, jitter bool) time.Duration {
	backoff := initial * time.Duration(1<<uint(attempt))
	if backoff > max {
		backoff = max
	}

	if jitter {
		// Add random jitter: ±25% of backoff duration
		jitterAmount := time.Duration(rand.Int63n(int64(backoff) / 2))
		if rand.Intn(2) == 0 {
			backoff += jitterAmount
		} else {
			backoff -= jitterAmount
		}
		if backoff < initial {
			backoff = initial
		}
	}

	return backoff
}

// IsRetryableHTTPError checks if an HTTP error status code is retryable.
func IsRetryableHTTPError(statusCode int) bool {
	// Retry on 5xx server errors and 429 (rate limit)
	return statusCode >= 500 || statusCode == 429
}

// IsRetryableNetworkError checks if a network error is retryable.
func IsRetryableNetworkError(err error) bool {
	if err == nil {
		return false
	}
	// Common retryable network errors
	errStr := err.Error()
	return containsAny(errStr, []string{
		"timeout",
		"connection refused",
		"connection reset",
		"no such host",
		"network is unreachable",
		"temporary failure",
		"i/o timeout",
	})
}

// containsAny checks if a string contains any of the substrings.
func containsAny(s string, substrings []string) bool {
	for _, substr := range substrings {
		if len(s) >= len(substr) {
			for i := 0; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
		}
	}
	return false
}

