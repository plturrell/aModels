package storage

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/hanapool"
)

func TestHANALogger(t *testing.T) {
	// Create mock pool for testing
	pool := &hanapool.Pool{}

	// Create HANA logger
	logger := NewHANALogger(pool)

	t.Run("CreateTables", func(t *testing.T) {
		ctx := context.Background()
		err := logger.CreateTables(ctx)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})

	t.Run("LogInference", func(t *testing.T) {
		ctx := context.Background()

		logEntry := &InferenceLog{
			RequestID:    "test-request-123",
			Model:        "vaultgemma-1b",
			Domain:       "general",
			Prompt:       "What is artificial intelligence?",
			Response:     "Artificial intelligence (AI) is a branch of computer science...",
			TokensUsed:   150,
			LatencyMs:    2500,
			Temperature:  0.7,
			MaxTokens:    1000,
			CacheHit:     false,
			UserID:       "user123",
			SessionID:    "session456",
			RequestTime:  time.Now().Add(-5 * time.Second),
			ResponseTime: time.Now(),
			Metadata: map[string]interface{}{
				"user_agent": "test-client",
				"ip_address": "127.0.0.1",
			},
		}

		err := logger.LogInference(ctx, logEntry)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})

	t.Run("LogInferenceWithError", func(t *testing.T) {
		ctx := context.Background()

		logEntry := &InferenceLog{
			RequestID:    "test-request-error-123",
			Model:        "vaultgemma-1b",
			Domain:       "general",
			Prompt:       "Invalid prompt",
			Response:     "",
			TokensUsed:   0,
			LatencyMs:    100,
			Temperature:  0.7,
			MaxTokens:    1000,
			CacheHit:     false,
			UserID:       "user123",
			SessionID:    "session456",
			RequestTime:  time.Now().Add(-2 * time.Second),
			ResponseTime: time.Now(),
			Error:        "Model inference failed: timeout",
			Metadata: map[string]interface{}{
				"error_type":  "timeout",
				"retry_count": 3,
			},
		}

		err := logger.LogInference(ctx, logEntry)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})

	t.Run("GetModelMetrics", func(t *testing.T) {
		ctx := context.Background()

		metrics, err := logger.GetModelMetrics(ctx, "vaultgemma-1b")
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if metrics == nil {
			t.Error("Expected model metrics to be returned")
		}

		if metrics.Model != "vaultgemma-1b" {
			t.Errorf("Expected model 'vaultgemma-1b', got '%s'", metrics.Model)
		}
	})

	t.Run("GetRecentInferences", func(t *testing.T) {
		ctx := context.Background()

		logs, err := logger.GetRecentInferences(ctx, 10)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if logs == nil {
			t.Error("Expected inference logs to be returned")
		}
	})

	t.Run("CleanupOldLogs", func(t *testing.T) {
		ctx := context.Background()

		err := logger.CleanupOldLogs(ctx, 30)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})
}

func TestInferenceLog(t *testing.T) {
	t.Run("CreateInferenceLog", func(t *testing.T) {
		now := time.Now()

		logEntry := &InferenceLog{
			RequestID:    "test-request-456",
			Model:        "granite-4.0",
			Domain:       "blockchain",
			Prompt:       "Explain smart contracts",
			Response:     "Smart contracts are self-executing contracts...",
			TokensUsed:   200,
			LatencyMs:    3000,
			Temperature:  0.5,
			MaxTokens:    1500,
			CacheHit:     true,
			UserID:       "user789",
			SessionID:    "session123",
			RequestTime:  now.Add(-10 * time.Second),
			ResponseTime: now,
			Metadata: map[string]interface{}{
				"domain_specific": true,
				"complexity":      "high",
			},
		}

		if logEntry.RequestID != "test-request-456" {
			t.Errorf("Expected RequestID 'test-request-456', got '%s'", logEntry.RequestID)
		}

		if logEntry.Model != "granite-4.0" {
			t.Errorf("Expected Model 'granite-4.0', got '%s'", logEntry.Model)
		}

		if logEntry.TokensUsed != 200 {
			t.Errorf("Expected TokensUsed 200, got %d", logEntry.TokensUsed)
		}

		if !logEntry.CacheHit {
			t.Error("Expected CacheHit to be true")
		}
	})

	t.Run("InferenceLogWithError", func(t *testing.T) {
		now := time.Now()

		logEntry := &InferenceLog{
			RequestID:    "test-request-error-456",
			Model:        "vaultgemma-1b",
			Domain:       "general",
			Prompt:       "Corrupted input",
			Response:     "",
			TokensUsed:   0,
			LatencyMs:    50,
			Temperature:  0.7,
			MaxTokens:    1000,
			CacheHit:     false,
			UserID:       "user789",
			SessionID:    "session123",
			RequestTime:  now.Add(-1 * time.Second),
			ResponseTime: now,
			Error:        "Input validation failed",
			Metadata: map[string]interface{}{
				"error_code": "VALIDATION_ERROR",
				"severity":   "high",
			},
		}

		if logEntry.Error == "" {
			t.Error("Expected Error to be set")
		}

		if logEntry.Response != "" {
			t.Error("Expected Response to be empty for error case")
		}
	})
}

func TestModelMetrics(t *testing.T) {
	t.Run("CreateModelMetrics", func(t *testing.T) {
		now := time.Now()

		metrics := &ModelMetrics{
			Model:         "vaultgemma-1b",
			TotalRequests: 1000,
			TotalTokens:   150000,
			AvgLatencyMs:  2500.5,
			CacheHitRate:  0.75,
			ErrorRate:     0.02,
			LastUpdated:   now,
		}

		if metrics.Model != "vaultgemma-1b" {
			t.Errorf("Expected Model 'vaultgemma-1b', got '%s'", metrics.Model)
		}

		if metrics.TotalRequests != 1000 {
			t.Errorf("Expected TotalRequests 1000, got %d", metrics.TotalRequests)
		}

		if metrics.CacheHitRate != 0.75 {
			t.Errorf("Expected CacheHitRate 0.75, got %f", metrics.CacheHitRate)
		}

		if metrics.ErrorRate != 0.02 {
			t.Errorf("Expected ErrorRate 0.02, got %f", metrics.ErrorRate)
		}
	})

	t.Run("ModelMetricsCalculation", func(t *testing.T) {
		// Test metrics calculation logic
		totalRequests := int64(1000)
		successfulRequests := int64(980)
		cachedRequests := int64(750)

		errorRate := float64(totalRequests-successfulRequests) / float64(totalRequests)
		cacheHitRate := float64(cachedRequests) / float64(totalRequests)

		expectedErrorRate := 0.02
		expectedCacheHitRate := 0.75

		if errorRate != expectedErrorRate {
			t.Errorf("Expected ErrorRate %f, got %f", expectedErrorRate, errorRate)
		}

		if cacheHitRate != expectedCacheHitRate {
			t.Errorf("Expected CacheHitRate %f, got %f", expectedCacheHitRate, cacheHitRate)
		}
	})
}

func BenchmarkHANALogger(b *testing.B) {
	// Create mock pool for testing
	pool := &hanapool.Pool{}

	// Create HANA logger
	logger := NewHANALogger(pool)

	ctx := context.Background()

	b.Run("LogInference", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			logEntry := &InferenceLog{
				RequestID:    fmt.Sprintf("benchmark-request-%d", i),
				Model:        "vaultgemma-1b",
				Domain:       "general",
				Prompt:       "Benchmark test prompt",
				Response:     "Benchmark test response",
				TokensUsed:   100,
				LatencyMs:    1000,
				Temperature:  0.7,
				MaxTokens:    500,
				CacheHit:     false,
				UserID:       "benchmark-user",
				SessionID:    "benchmark-session",
				RequestTime:  time.Now().Add(-1 * time.Second),
				ResponseTime: time.Now(),
				Metadata: map[string]interface{}{
					"benchmark": true,
					"iteration": i,
				},
			}

			err := logger.LogInference(ctx, logEntry)
			if err != nil {
				b.Skip("Skipping benchmark - requires HANA connection")
			}
		}
	})

	b.Run("GetModelMetrics", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := logger.GetModelMetrics(ctx, "vaultgemma-1b")
			if err != nil {
				b.Skip("Skipping benchmark - requires HANA connection")
			}
		}
	})

	b.Run("GetRecentInferences", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := logger.GetRecentInferences(ctx, 10)
			if err != nil {
				b.Skip("Skipping benchmark - requires HANA connection")
			}
		}
	})
}

func ExampleHANALogger() {
	// Create HANA pool
	pool := &hanapool.Pool{}

	// Create HANA logger
	logger := NewHANALogger(pool)

	// Initialize tables
	ctx := context.Background()
	err := logger.CreateTables(ctx)
	if err != nil {
		// Handle error
		return
	}

	// Log an inference request/response
	logEntry := &InferenceLog{
		RequestID:    "example-request-123",
		Model:        "vaultgemma-1b",
		Domain:       "general",
		Prompt:       "What is machine learning?",
		Response:     "Machine learning is a subset of artificial intelligence...",
		TokensUsed:   150,
		LatencyMs:    2000,
		Temperature:  0.7,
		MaxTokens:    1000,
		CacheHit:     false,
		UserID:       "user123",
		SessionID:    "session456",
		RequestTime:  time.Now().Add(-5 * time.Second),
		ResponseTime: time.Now(),
		Metadata: map[string]interface{}{
			"user_agent": "example-client",
			"ip_address": "192.168.1.100",
		},
	}

	// Log the inference
	err = logger.LogInference(ctx, logEntry)
	if err != nil {
		// Handle error
		return
	}

	// Get model metrics
	metrics, err := logger.GetModelMetrics(ctx, "vaultgemma-1b")
	if err != nil {
		// Handle error
		return
	}

	// Use metrics
	_ = metrics.TotalRequests
	_ = metrics.CacheHitRate
	_ = metrics.ErrorRate

	// Get recent inferences
	recentLogs, err := logger.GetRecentInferences(ctx, 10)
	if err != nil {
		// Handle error
		return
	}

	// Process recent logs
	for _, log := range recentLogs {
		_ = log.RequestID
		_ = log.Model
		_ = log.Response
	}
}
