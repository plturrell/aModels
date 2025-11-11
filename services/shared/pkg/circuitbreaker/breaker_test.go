package circuitbreaker

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/sony/gobreaker"
)

func TestCircuitBreaker_Execute(t *testing.T) {
	config := DefaultConfig("test-breaker")
	cb := New(config)

	// Successful execution
	result, err := cb.Execute(func() (interface{}, error) {
		return "success", nil
	})

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if result != "success" {
		t.Errorf("Expected 'success', got %v", result)
	}
	if cb.State() != gobreaker.StateClosed {
		t.Errorf("Expected closed state, got %v", cb.State())
	}
}

func TestCircuitBreaker_Failures(t *testing.T) {
	config := DefaultConfig("test-breaker")
	config.ReadyToTrip = func(counts gobreaker.Counts) bool {
		return counts.ConsecutiveFailures > 3
	}
	cb := New(config)

	// Cause failures
	for i := 0; i < 4; i++ {
		_, err := cb.Execute(func() (interface{}, error) {
			return nil, errors.New("failure")
		})
		if err == nil {
			t.Error("Expected error")
		}
	}

	// Circuit should be open now
	if cb.State() != gobreaker.StateOpen {
		t.Errorf("Expected open state, got %v", cb.State())
	}

	// Should fail fast when open
	_, err := cb.Execute(func() (interface{}, error) {
		return "should not execute", nil
	})
	if err != gobreaker.ErrOpenState {
		t.Errorf("Expected ErrOpenState, got %v", err)
	}
}

func TestCircuitBreaker_ExecuteWithContext(t *testing.T) {
	config := DefaultConfig("test-breaker")
	cb := New(config)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := cb.ExecuteWithContext(ctx, func() (interface{}, error) {
		return "should not execute", nil
	})

	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
}

func TestCircuitBreaker_StateTransitions(t *testing.T) {
	config := DefaultConfig("test-breaker")
	config.Timeout = 100 * time.Millisecond
	config.ReadyToTrip = func(counts gobreaker.Counts) bool {
		return counts.ConsecutiveFailures > 2
	}
	cb := New(config)

	// Start closed
	if !cb.IsClosed() {
		t.Error("Expected closed state initially")
	}

	// Cause failures to open
	for i := 0; i < 3; i++ {
		cb.Execute(func() (interface{}, error) {
			return nil, errors.New("failure")
		})
	}

	if !cb.IsOpen() {
		t.Error("Expected open state after failures")
	}

	// Wait for timeout
	time.Sleep(150 * time.Millisecond)

	// Try to execute (should transition to half-open)
	cb.Execute(func() (interface{}, error) {
		return "success", nil
	})

	// Should be closed again after success
	if !cb.IsClosed() {
		t.Error("Expected closed state after recovery")
	}
}

func TestCircuitBreaker_Counts(t *testing.T) {
	config := DefaultConfig("test-breaker")
	cb := New(config)

	// Execute some operations
	cb.Execute(func() (interface{}, error) {
		return "success", nil
	})
	cb.Execute(func() (interface{}, error) {
		return nil, errors.New("failure")
	})

	counts := cb.Counts()
	if counts.Requests != 2 {
		t.Errorf("Expected 2 requests, got %d", counts.Requests)
	}
	if counts.TotalSuccesses != 1 {
		t.Errorf("Expected 1 success, got %d", counts.TotalSuccesses)
	}
	if counts.TotalFailures != 1 {
		t.Errorf("Expected 1 failure, got %d", counts.TotalFailures)
	}
}

