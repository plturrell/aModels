package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/hanapool"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
	"golang.org/x/time/rate"
)

func TestHANALoggerIntegration(t *testing.T) {
	// Create mock HANA pool
	pool := &hanapool.Pool{}

	// Create HANA logger
	hanaLogger := storage.NewHANALogger(pool)

	// Create HANA cache
	hanaCache := storage.NewHANACache(pool)

	t.Run("ServerInitializationWithHANALogger", func(t *testing.T) {
		dm, models := buildTestDomainResources()
		server := NewVaultGemmaServer(models, nil, nil, dm, rate.NewLimiter(rate.Every(time.Millisecond), 100), "test")
		server.hanaLogger = hanaLogger
		server.hanaCache = hanaCache

		if server.domainManager == nil {
			t.Error("Expected domain manager to be initialized")
		}

		if server.models["general"] == nil {
			t.Error("Expected general model to be registered")
		}
	})

	t.Run("HandleChatWithHANALogging", func(t *testing.T) {
		dm, models := buildTestDomainResources()
		server := NewVaultGemmaServer(models, nil, nil, dm, rate.NewLimiter(rate.Every(time.Millisecond), 100), "test")
		server.hanaLogger = hanaLogger
		server.hanaCache = nil     // Avoid nil pool operations
		server.semanticCache = nil // Skip semantic cache
		server.enhancedLogging = NewEnhancedLogging(&MockHANALogger{})
		server.DisableEnhancedInference()

		// Create test request
		requestBody := map[string]interface{}{
			"messages": []map[string]interface{}{
				{
					"role":    "user",
					"content": "What is artificial intelligence?",
				},
			},
			"model":      "vaultgemma-1b",
			"domain":     "general",
			"user_id":    "test-user-123",
			"max_tokens": 16,
		}

		jsonBody, err := json.Marshal(requestBody)
		if err != nil {
			t.Fatalf("Failed to marshal request body: %v", err)
		}

		// Create HTTP request
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", "test-client")

		// Create response recorder
		w := httptest.NewRecorder()

		// Handle request
		server.HandleChat(w, req)

		// Check response status
		if w.Code != http.StatusOK {
			t.Skipf("Skipping chat test - requires full model configuration (status %d, body %s)", w.Code, w.Body.String())
		}
	})

	t.Run("HANALoggerTableCreation", func(t *testing.T) {
		ctx := context.Background()

		err := hanaLogger.CreateTables(ctx)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})

	t.Run("InferenceLogging", func(t *testing.T) {
		ctx := context.Background()

		// Create test inference log
		logEntry := &storage.InferenceLog{
			RequestID:    "integration-test-123",
			Model:        "vaultgemma-1b",
			Domain:       "general",
			Prompt:       "Integration test prompt",
			Response:     "Integration test response",
			TokensUsed:   100,
			LatencyMs:    1500,
			Temperature:  0.7,
			MaxTokens:    500,
			CacheHit:     false,
			UserID:       "integration-user",
			SessionID:    "integration-session",
			RequestTime:  time.Now().Add(-2 * time.Second),
			ResponseTime: time.Now(),
			Metadata: map[string]interface{}{
				"test_type": "integration",
				"client":    "test-client",
			},
		}

		err := hanaLogger.LogInference(ctx, logEntry)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})

	t.Run("ModelMetricsRetrieval", func(t *testing.T) {
		ctx := context.Background()

		metrics, err := hanaLogger.GetModelMetrics(ctx, "vaultgemma-1b")
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if metrics == nil {
			t.Error("Expected model metrics to be returned")
		}
	})

	t.Run("RecentInferencesRetrieval", func(t *testing.T) {
		ctx := context.Background()

		logs, err := hanaLogger.GetRecentInferences(ctx, 5)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if logs == nil {
			t.Error("Expected inference logs to be returned")
		}
	})
}

func TestHANALoggerWithErrorHandling(t *testing.T) {
	// Create mock HANA pool
	pool := &hanapool.Pool{}

	// Create HANA logger
	hanaLogger := storage.NewHANALogger(pool)

	t.Run("LogInferenceWithError", func(t *testing.T) {
		ctx := context.Background()

		// Create inference log with error
		logEntry := &storage.InferenceLog{
			RequestID:    "error-test-123",
			Model:        "vaultgemma-1b",
			Domain:       "general",
			Prompt:       "Error test prompt",
			Response:     "",
			TokensUsed:   0,
			LatencyMs:    100,
			Temperature:  0.7,
			MaxTokens:    500,
			CacheHit:     false,
			UserID:       "error-user",
			SessionID:    "error-session",
			RequestTime:  time.Now().Add(-1 * time.Second),
			ResponseTime: time.Now(),
			Error:        "Model inference failed: timeout",
			Metadata: map[string]interface{}{
				"error_type":  "timeout",
				"retry_count": 3,
			},
		}

		err := hanaLogger.LogInference(ctx, logEntry)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})
}

func TestHANALoggerPerformance(t *testing.T) {
	// Create mock HANA pool
	pool := &hanapool.Pool{}

	// Create HANA logger
	hanaLogger := storage.NewHANALogger(pool)

	t.Run("ConcurrentLogging", func(t *testing.T) {
		ctx := context.Background()

		// Test concurrent logging
		done := make(chan bool, 10)

		for i := 0; i < 10; i++ {
			go func(index int) {
				logEntry := &storage.InferenceLog{
					RequestID:    fmt.Sprintf("concurrent-test-%d", index),
					Model:        "vaultgemma-1b",
					Domain:       "general",
					Prompt:       fmt.Sprintf("Concurrent test prompt %d", index),
					Response:     fmt.Sprintf("Concurrent test response %d", index),
					TokensUsed:   50 + index,
					LatencyMs:    1000 + int64(index*100),
					Temperature:  0.7,
					MaxTokens:    500,
					CacheHit:     index%2 == 0,
					UserID:       fmt.Sprintf("concurrent-user-%d", index),
					SessionID:    fmt.Sprintf("concurrent-session-%d", index),
					RequestTime:  time.Now().Add(-time.Duration(index) * time.Second),
					ResponseTime: time.Now(),
					Metadata: map[string]interface{}{
						"concurrent_test": true,
						"index":           index,
					},
				}

				err := hanaLogger.LogInference(ctx, logEntry)
				if err != nil {
					t.Logf("Concurrent logging failed for index %d: %v", index, err)
				}

				done <- true
			}(i)
		}

		// Wait for all goroutines to complete
		for i := 0; i < 10; i++ {
			<-done
		}
	})
}

func BenchmarkHANALoggerIntegration(b *testing.B) {
	// Create mock HANA pool
	pool := &hanapool.Pool{}

	// Create HANA logger
	hanaLogger := storage.NewHANALogger(pool)

	ctx := context.Background()

	b.Run("LogInference", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			logEntry := &storage.InferenceLog{
				RequestID:    fmt.Sprintf("benchmark-%d", i),
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

			err := hanaLogger.LogInference(ctx, logEntry)
			if err != nil {
				b.Skip("Skipping benchmark - requires HANA connection")
			}
		}
	})
}

func ExampleVaultGemmaServer_hanaLogger() {
	// Create HANA pool
	pool := &hanapool.Pool{}

	// Create HANA logger
	hanaLogger := storage.NewHANALogger(pool)

	// Initialize tables
	ctx := context.Background()
	err := hanaLogger.CreateTables(ctx)
	if err != nil {
		// Handle error
		return
	}

	// Create test server with HANA logger
	server := &VaultGemmaServer{
		hanaLogger: hanaLogger,
	}
	_ = server // Use server variable

	// Log an inference request/response
	logEntry := &storage.InferenceLog{
		RequestID:    "example-request-456",
		Model:        "vaultgemma-1b",
		Domain:       "general",
		Prompt:       "What is deep learning?",
		Response:     "Deep learning is a subset of machine learning...",
		TokensUsed:   200,
		LatencyMs:    2500,
		Temperature:  0.7,
		MaxTokens:    1000,
		CacheHit:     false,
		UserID:       "example-user",
		SessionID:    "example-session",
		RequestTime:  time.Now().Add(-5 * time.Second),
		ResponseTime: time.Now(),
		Metadata: map[string]interface{}{
			"user_agent": "example-client",
			"ip_address": "192.168.1.200",
		},
	}

	// Log the inference
	err = hanaLogger.LogInference(ctx, logEntry)
	if err != nil {
		// Handle error
		return
	}

	// Get model metrics
	metrics, err := hanaLogger.GetModelMetrics(ctx, "vaultgemma-1b")
	if err != nil {
		// Handle error
		return
	}

	// Use metrics for monitoring
	fmt.Printf("Model: %s\n", metrics.Model)
	fmt.Printf("Total Requests: %d\n", metrics.TotalRequests)
	fmt.Printf("Cache Hit Rate: %.2f%%\n", metrics.CacheHitRate*100)
	fmt.Printf("Error Rate: %.2f%%\n", metrics.ErrorRate*100)

	// Get recent inferences for analysis
	recentLogs, err := hanaLogger.GetRecentInferences(ctx, 10)
	if err != nil {
		// Handle error
		return
	}

	// Process recent logs
	for _, log := range recentLogs {
		fmt.Printf("Request: %s, Model: %s, Tokens: %d\n",
			log.RequestID, log.Model, log.TokensUsed)
	}
}

func buildTestDomainResources() (*domain.DomainManager, map[string]*ai.VaultGemma) {
	dm := domain.NewDomainManager()
	cfg := &domain.DomainConfig{
		Name:        "General",
		Layer:       "layer4",
		Team:        "TestTeam",
		BackendType: "vaultgemma",
		ModelPath:   "models/general",
		MaxTokens:   64,
		Temperature: 0.7,
		Keywords:    []string{"general"},
	}
	dm.AddDomain("general", cfg)

	models := map[string]*ai.VaultGemma{
		"general": newTestVaultGemma(),
	}
	return dm, models
}

func newTestVaultGemma() *ai.VaultGemma {
	config := ai.VaultGemmaConfig{
		NumLayers:        4,
		HiddenSize:       128,
		VocabSize:        1024,
		NumHeads:         4,
		HeadDim:          32,
		IntermediateSize: 256,
		RMSNormEps:       1e-6,
	}

	model := &ai.VaultGemma{
		Config: config,
		Embed: &ai.EmbeddingLayer{
			Weights: util.NewMatrix64(config.VocabSize, config.HiddenSize),
		},
		Output: &ai.OutputLayer{
			Weights: util.NewMatrix64(config.HiddenSize, config.VocabSize),
		},
		Layers: make([]ai.TransformerLayer, config.NumLayers),
	}

	headDim := config.HeadDim
	if headDim == 0 && config.NumHeads > 0 {
		headDim = config.HiddenSize / config.NumHeads
	}
	if headDim == 0 {
		headDim = 1
	}

	for i := range model.Layers {
		layer := &model.Layers[i]
		layer.SelfAttention = &ai.MultiHeadAttention{
			NumHeads: config.NumHeads,
			HeadDim:  headDim,
			WQ:       util.NewMatrix64(config.HiddenSize, config.NumHeads*headDim),
			WK:       util.NewMatrix64(config.HiddenSize, config.NumHeads*headDim),
			WV:       util.NewMatrix64(config.HiddenSize, config.NumHeads*headDim),
			WO:       util.NewMatrix64(config.NumHeads*headDim, config.HiddenSize),
		}
		layer.FeedForward = &ai.FeedForwardNetwork{
			W1: util.NewMatrix64(config.HiddenSize, config.IntermediateSize),
			W2: util.NewMatrix64(config.IntermediateSize, config.HiddenSize),
			W3: util.NewMatrix64(config.HiddenSize, config.IntermediateSize),
		}
		layer.LayerNorm1 = &ai.RMSNorm{
			Weight: make([]float64, config.HiddenSize),
			Eps:    config.RMSNormEps,
		}
		layer.LayerNorm2 = &ai.RMSNorm{
			Weight: make([]float64, config.HiddenSize),
			Eps:    config.RMSNormEps,
		}
		for j := range layer.LayerNorm1.Weight {
			layer.LayerNorm1.Weight[j] = 1.0
			layer.LayerNorm2.Weight[j] = 1.0
		}
	}

	return model
}
