package utils

import (
	"context"
	"fmt"
	"log"
	"math"
	"time"
)

// RetryConfig configures retry behavior
type RetryConfig struct {
	MaxAttempts      int
	InitialBackoff   time.Duration
	MaxBackoff       time.Duration
	BackoffMultiplier float64
	RetryableErrors  []error
}

// DefaultRetryConfig returns a default retry configuration
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:       3,
		InitialBackoff:    100 * time.Millisecond,
		MaxBackoff:        5 * time.Second,
		BackoffMultiplier: 2.0,
		RetryableErrors:   []error{},
	}
}

// RetryMetrics tracks retry statistics
type RetryMetrics struct {
	TotalAttempts   int
	SuccessfulRetries int
	FailedRetries   int
	TotalRetryTime  time.Duration
}

// RetryableFunc is a function that can be retried
type RetryableFunc func() error

// RetryWithBackoff executes a function with exponential backoff retry logic
func RetryWithBackoff(ctx context.Context, fn RetryableFunc, config RetryConfig, logger *log.Logger) error {
	var lastErr error
	metrics := RetryMetrics{}

	for attempt := 0; attempt < config.MaxAttempts; attempt++ {
		metrics.TotalAttempts++

		// Check context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Execute the function
		startTime := time.Now()
		err := fn()
		metrics.TotalRetryTime += time.Since(startTime)

		if err == nil {
			if attempt > 0 && logger != nil {
				logger.Printf("Retry succeeded after %d attempts", attempt+1)
				metrics.SuccessfulRetries++
			}
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !isRetryableError(err, config.RetryableErrors) {
			if logger != nil {
				logger.Printf("Non-retryable error: %v", err)
			}
			return err
		}

		// Don't sleep after the last attempt
		if attempt < config.MaxAttempts-1 {
			backoff := calculateBackoff(attempt, config)
			
			if logger != nil {
				logger.Printf("Attempt %d/%d failed: %v, retrying in %v", attempt+1, config.MaxAttempts, err, backoff)
			}

			// Sleep with context cancellation support
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
				// Continue to next attempt
			}
		}
	}

	metrics.FailedRetries++
	if logger != nil {
		logger.Printf("All %d retry attempts failed. Last error: %v", config.MaxAttempts, lastErr)
	}

	return fmt.Errorf("retry failed after %d attempts: %w", config.MaxAttempts, lastErr)
}

// calculateBackoff calculates the backoff duration for the given attempt
func calculateBackoff(attempt int, config RetryConfig) time.Duration {
	backoff := float64(config.InitialBackoff) * math.Pow(config.BackoffMultiplier, float64(attempt))
	maxBackoff := float64(config.MaxBackoff)
	
	if backoff > maxBackoff {
		backoff = maxBackoff
	}

	return time.Duration(backoff)
}

// isRetryableError checks if an error is retryable
func isRetryableError(err error, retryableErrors []error) bool {
	if err == nil {
		return false
	}

	// If no specific retryable errors are configured, retry all errors
	if len(retryableErrors) == 0 {
		return true
	}

	// Check if error matches any retryable error
	for _, retryableErr := range retryableErrors {
		if err == retryableErr || fmt.Sprintf("%T", err) == fmt.Sprintf("%T", retryableErr) {
			return true
		}
	}

	return false
}

// RetryPostgresOperation wraps a Postgres operation with retry logic
func RetryPostgresOperation(ctx context.Context, fn RetryableFunc, logger *log.Logger) error {
	config := DefaultRetryConfig()
	config.MaxAttempts = 3
	config.InitialBackoff = 200 * time.Millisecond
	return RetryWithBackoff(ctx, fn, config, logger)
}

// RetryRedisOperation wraps a Redis operation with retry logic
func RetryRedisOperation(ctx context.Context, fn RetryableFunc, logger *log.Logger) error {
	config := DefaultRetryConfig()
	config.MaxAttempts = 3
	config.InitialBackoff = 100 * time.Millisecond
	return RetryWithBackoff(ctx, fn, config, logger)
}

// RetryNeo4jOperation wraps a Neo4j operation with retry logic
func RetryNeo4jOperation(ctx context.Context, fn RetryableFunc, logger *log.Logger) error {
	config := DefaultRetryConfig()
	config.MaxAttempts = 3
	config.InitialBackoff = 300 * time.Millisecond
	return RetryWithBackoff(ctx, fn, config, logger)
}

