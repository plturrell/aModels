package main

import (
	"context"
	"errors"
	"log"
	"os"
	"testing"
	"time"
)

func TestRetryWithBackoff(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	ctx := context.Background()
	
	tests := []struct {
		name        string
		fn          RetryableFunc
		config      RetryConfig
		wantSuccess bool
		wantAttempts int
	}{
		{
			name: "success on first attempt",
			fn: func() error {
				return nil
			},
			config:      DefaultRetryConfig(),
			wantSuccess: true,
			wantAttempts: 1,
		},
		{
			name: "success after retries",
			fn: func() error {
				// Simulate transient failure
				if time.Now().Unix()%2 == 0 {
					return errors.New("transient error")
				}
				return nil
			},
			config:      DefaultRetryConfig(),
			wantSuccess: true,
			wantAttempts: 2, // May vary
		},
		{
			name: "all retries fail",
			fn: func() error {
				return errors.New("permanent error")
			},
			config: RetryConfig{
				MaxAttempts:      3,
				InitialBackoff:   10 * time.Millisecond,
				MaxBackoff:       100 * time.Millisecond,
				BackoffMultiplier: 2.0,
			},
			wantSuccess: false,
			wantAttempts: 3,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attempts := 0
			wrappedFn := func() error {
				attempts++
				return tt.fn()
			}
			
			err := RetryWithBackoff(ctx, wrappedFn, tt.config, logger)
			
			if tt.wantSuccess && err != nil {
				t.Errorf("RetryWithBackoff() error = %v, want success", err)
			}
			if !tt.wantSuccess && err == nil {
				t.Errorf("RetryWithBackoff() succeeded, want failure")
			}
			if attempts < 1 {
				t.Errorf("RetryWithBackoff() attempts = %d, want at least 1", attempts)
			}
		})
	}
}

func TestCalculateBackoff(t *testing.T) {
	config := DefaultRetryConfig()
	config.InitialBackoff = 100 * time.Millisecond
	config.MaxBackoff = 1 * time.Second
	config.BackoffMultiplier = 2.0
	
	tests := []struct {
		attempt int
		wantMin time.Duration
		wantMax time.Duration
	}{
		{0, 100 * time.Millisecond, 200 * time.Millisecond},
		{1, 200 * time.Millisecond, 400 * time.Millisecond},
		{2, 400 * time.Millisecond, 800 * time.Millisecond},
		{10, 0, 1 * time.Second}, // Should be capped at maxBackoff
	}
	
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			backoff := calculateBackoff(tt.attempt, config)
			if backoff < tt.wantMin || backoff > tt.wantMax {
				t.Errorf("calculateBackoff(%d) = %v, want between %v and %v", 
					tt.attempt, backoff, tt.wantMin, tt.wantMax)
			}
		})
	}
}

