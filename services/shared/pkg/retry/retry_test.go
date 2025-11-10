package retry

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestWithRetry_Success(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.MaxAttempts = 3

	attempts := 0
	err := WithRetry(ctx, config, func() error {
		attempts++
		return nil
	})

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if attempts != 1 {
		t.Errorf("Expected 1 attempt, got %d", attempts)
	}
}

func TestWithRetry_RetryableError(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.MaxAttempts = 3
	config.InitialBackoff = 10 * time.Millisecond

	attempts := 0
	err := WithRetry(ctx, config, func() error {
		attempts++
		if attempts < 3 {
			return errors.New("retryable error")
		}
		return nil
	})

	if err != nil {
		t.Fatalf("Expected no error after retries, got %v", err)
	}
	if attempts != 3 {
		t.Errorf("Expected 3 attempts, got %d", attempts)
	}
}

func TestWithRetry_NonRetryableError(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.MaxAttempts = 3
	config.Retryable = func(err error) bool {
		return false // Don't retry any errors
	}

	attempts := 0
	err := WithRetry(ctx, config, func() error {
		attempts++
		return errors.New("non-retryable error")
	})

	if err == nil {
		t.Fatal("Expected error, got nil")
	}
	if attempts != 1 {
		t.Errorf("Expected 1 attempt (no retry), got %d", attempts)
	}
}

func TestWithRetry_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	config := DefaultConfig()
	config.MaxAttempts = 5
	config.InitialBackoff = 100 * time.Millisecond

	attempts := 0
	errChan := make(chan error, 1)

	go func() {
		err := WithRetry(ctx, config, func() error {
			attempts++
			if attempts == 1 {
				cancel() // Cancel after first attempt
			}
			return errors.New("retryable error")
		})
		errChan <- err
	}()

	err := <-errChan
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
	if attempts > 2 {
		t.Errorf("Expected at most 2 attempts, got %d", attempts)
	}
}

func TestCalculateBackoff(t *testing.T) {
	initial := 100 * time.Millisecond
	max := 5 * time.Second

	// Test exponential backoff
	backoff1 := calculateBackoff(1, initial, max, false)
	if backoff1 != 200*time.Millisecond {
		t.Errorf("Expected 200ms, got %v", backoff1)
	}

	backoff2 := calculateBackoff(2, initial, max, false)
	if backoff2 != 400*time.Millisecond {
		t.Errorf("Expected 400ms, got %v", backoff2)
	}

	// Test max cap
	backoff10 := calculateBackoff(10, initial, max, false)
	if backoff10 != max {
		t.Errorf("Expected %v, got %v", max, backoff10)
	}
}

func TestIsRetryableHTTPError(t *testing.T) {
	tests := []struct {
		statusCode int
		retryable  bool
	}{
		{200, false},
		{400, false},
		{429, true}, // Rate limit
		{500, true}, // Server error
		{503, true}, // Service unavailable
	}

	for _, tt := range tests {
		result := IsRetryableHTTPError(tt.statusCode)
		if result != tt.retryable {
			t.Errorf("Status %d: expected %v, got %v", tt.statusCode, tt.retryable, result)
		}
	}
}

