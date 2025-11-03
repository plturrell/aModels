package server

import (
	"context"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
)

// DefaultRetryConfig returns default retry configuration
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   100 * time.Millisecond,
		MaxDelay:    5 * time.Second,
		Multiplier:  2.0,
		Jitter:      true,
	}
}

// RetryableError represents an error that can be retried
type RetryableError struct {
	Err       error
	Retryable bool
}

func (e *RetryableError) Error() string {
	return e.Err.Error()
}

func (e *RetryableError) Unwrap() error {
	return e.Err
}

// IsRetryableError checks if an error is retryable
func IsRetryableError(err error) bool {
	var retryableErr *RetryableError
	return err != nil && (retryableErr == nil || retryableErr.Retryable)
}

// RetryFunc represents a function that can be retried
type RetryFunc func(ctx context.Context) (interface{}, error)

// Retry executes a function with retry logic
func (s *VaultGemmaServer) Retry(ctx context.Context, config *RetryConfig, fn RetryFunc) (interface{}, error) {
	if config == nil {
		config = DefaultRetryConfig()
	}

	var lastErr error
	attempt := 0

	for attempt < config.MaxAttempts {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		result, err := fn(ctx)
		if err == nil {
			if attempt > 0 {
				log.Printf("Retry succeeded on attempt %d", attempt+1)
			}
			return result, nil
		}

		lastErr = err

		// Check if error is retryable
		if !IsRetryableError(err) {
			log.Printf("Non-retryable error: %v", err)
			return nil, err
		}

		attempt++
		if attempt >= config.MaxAttempts {
			log.Printf("Max retry attempts (%d) exceeded", config.MaxAttempts)
			break
		}

		// Calculate delay with exponential backoff
		delay := s.calculateDelay(config, attempt)
		log.Printf("Retry attempt %d failed: %v, retrying in %v", attempt, err, delay)

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
			// Continue to next attempt
		}
	}

	return nil, fmt.Errorf("retry failed after %d attempts: %w", config.MaxAttempts, lastErr)
}

// calculateDelay calculates the delay for the next retry attempt
func (s *VaultGemmaServer) calculateDelay(config *RetryConfig, attempt int) time.Duration {
	// Exponential backoff: delay = baseDelay * (multiplier ^ attempt)
	delay := float64(config.BaseDelay) * math.Pow(config.Multiplier, float64(attempt-1))

	// Apply jitter if enabled (randomize delay by ±25%)
	if config.Jitter {
		jitter := 0.25
		// Simple jitter using time-based seed
		seed := time.Now().UnixNano()
		jitterFactor := 1.0 - jitter + (float64(seed%1000)/1000.0)*2*jitter
		delay *= jitterFactor
	}

	// Cap at max delay
	if delay > float64(config.MaxDelay) {
		delay = float64(config.MaxDelay)
	}

	return time.Duration(delay)
}

// RetryableInferenceRequest represents a retryable inference request
type RetryableInferenceRequest struct {
	Prompt      string
	Domain      string
	MaxTokens   int
	Temperature float64
	Model       interface{} // *ai.VaultGemma
	TopP        float64
	TopK        int
}

// RetryableInferenceResponse represents a retryable inference response
type RetryableInferenceResponse struct {
	Content    string
	TokensUsed int
	Error      error
}

// RetryInference performs inference with retry logic
func (s *VaultGemmaServer) RetryInference(ctx context.Context, config *RetryConfig, req *RetryableInferenceRequest) (*RetryableInferenceResponse, error) {
	fn := func(ctx context.Context) (interface{}, error) {
		// Use enhanced inference engine if available
		if s.enhancedEngine != nil {
			enhancedReq := &inference.EnhancedInferenceRequest{
				Prompt:      req.Prompt,
				Domain:      req.Domain,
				MaxTokens:   req.MaxTokens,
				Temperature: req.Temperature,
				Model:       req.Model.(*ai.VaultGemma),
				TopP:        req.TopP,
				TopK:        req.TopK,
			}

			response := s.enhancedEngine.GenerateEnhancedResponse(ctx, enhancedReq)
			if response.Error != nil {
				// Check if error is retryable
				if s.isRetryableInferenceError(response.Error) {
					return nil, &RetryableError{
						Err:       response.Error,
						Retryable: true,
					}
				}
				return nil, response.Error
			}

			return &RetryableInferenceResponse{
				Content:    response.Content,
				TokensUsed: response.TokensUsed,
				Error:      nil,
			}, nil
		} else {
			// Fall back to basic inference engine
			if s.inferenceEngine == nil {
				s.inferenceEngine = inference.NewInferenceEngine(s.models, s.domainManager)
			}

			inferenceReq := &inference.InferenceRequest{
				Prompt:      req.Prompt,
				Domain:      req.Domain,
				MaxTokens:   req.MaxTokens,
				Temperature: req.Temperature,
				Model:       req.Model.(*ai.VaultGemma),
				TopP:        req.TopP,
				TopK:        req.TopK,
			}

			response := s.inferenceEngine.GenerateResponse(ctx, inferenceReq)
			if response.Error != nil {
				// Check if error is retryable
				if s.isRetryableInferenceError(response.Error) {
					return nil, &RetryableError{
						Err:       response.Error,
						Retryable: true,
					}
				}
				return nil, response.Error
			}

			return &RetryableInferenceResponse{
				Content:    response.Content,
				TokensUsed: response.TokensUsed,
				Error:      nil,
			}, nil
		}
	}

	result, err := s.Retry(ctx, config, fn)
	if err != nil {
		return nil, err
	}

	return result.(*RetryableInferenceResponse), nil
}

// isRetryableInferenceError checks if an inference error is retryable
func (s *VaultGemmaServer) isRetryableInferenceError(err error) bool {
	if err == nil {
		return false
	}

	// Check for common retryable errors
	errorStr := err.Error()
	retryableErrors := []string{
		"timeout",
		"connection",
		"network",
		"temporary",
		"rate limit",
		"throttle",
		"busy",
		"overloaded",
		"resource",
		"memory",
		"context deadline",
	}

	for _, retryableErr := range retryableErrors {
		if contains(errorStr, retryableErr) {
			return true
		}
	}

	return false
}

// contains checks if a string contains a substring (case-insensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr ||
			(len(s) > len(substr) &&
				(s[:len(substr)] == substr ||
					s[len(s)-len(substr):] == substr ||
					containsSubstring(s, substr))))
}

// containsSubstring checks if a string contains a substring
func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// RetryableCacheOperation represents a retryable cache operation
type RetryableCacheOperation struct {
	Operation string
	Key       string
	Value     interface{}
	TTL       time.Duration
}

// RetryCacheOperation performs cache operations with retry logic
func (s *VaultGemmaServer) RetryCacheOperation(ctx context.Context, config *RetryConfig, op *RetryableCacheOperation) error {
	if s.hanaCache == nil {
		return fmt.Errorf("cache not available")
	}

	fn := func(ctx context.Context) (interface{}, error) {
		switch op.Operation {
		case "get":
			_, err := s.hanaCache.Get(ctx, op.Key)
			return nil, err
		case "set":
			if cacheEntry, ok := op.Value.(*storage.CacheEntry); ok {
				err := s.hanaCache.Set(ctx, cacheEntry)
				return nil, err
			}
			return nil, fmt.Errorf("invalid cache entry type")
		case "delete":
			// HANACache doesn't have Delete method, skip for now
			return nil, fmt.Errorf("delete operation not supported")
		default:
			return nil, fmt.Errorf("unknown cache operation: %s", op.Operation)
		}
	}

	_, err := s.Retry(ctx, config, fn)
	return err
}

// RetryableLogOperation represents a retryable logging operation
type RetryableLogOperation struct {
	LogEntry interface{}
}

// RetryLogOperation performs logging operations with retry logic
func (s *VaultGemmaServer) RetryLogOperation(ctx context.Context, config *RetryConfig, op *RetryableLogOperation) error {
	if s.hanaLogger == nil {
		return fmt.Errorf("logger not available")
	}

	fn := func(ctx context.Context) (interface{}, error) {
		// This is a simplified version - in practice, you'd need to handle different log entry types
		return nil, fmt.Errorf("logging operation not implemented")
	}

	_, err := s.Retry(ctx, config, fn)
	return err
}
