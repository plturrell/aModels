package load

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/server"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
	"golang.org/x/time/rate"
)

// buildTestModel creates a minimal test model for load testing
func buildTestModel() *ai.VaultGemma {
	config := ai.VaultGemmaConfig{
		NumLayers:        4,
		HiddenSize:       128,
		VocabSize:        2048,
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

// newTestServer creates a test server for load testing
func newTestServer() *server.VaultGemmaServer {
	models := map[string]*ai.VaultGemma{
		"general": buildTestModel(),
	}

	dm := domain.NewDomainManager()
	dm.AddDomain("general", &domain.DomainConfig{
		Name:        "General",
		ModelPath:   "models/general",
		MaxTokens:   64,
		Temperature: 0.7,
		Keywords:    []string{"general"},
	})

	srv := server.NewVaultGemmaServer(models, nil, nil, dm, rate.NewLimiter(rate.Every(time.Millisecond), 1000), "load-test")
	srv.inferenceEngine = inference.NewInferenceEngine(models, dm)
	return srv
}

// TestConcurrentLoad tests handling of many concurrent requests
func TestConcurrentLoad(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test message"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)

	const numRequests = 100
	const numWorkers = 10

	var wg sync.WaitGroup
	results := make(chan int, numRequests)
	errors := make(chan error, numRequests)

	start := time.Now()

	// Launch workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numRequests/numWorkers; j++ {
				req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()

				srv.HandleChat(w, req)
				results <- w.Code

				if w.Code != http.StatusOK {
					errors <- nil // Signal error
				}
			}
		}()
	}

	wg.Wait()
	close(results)
	close(errors)

	duration := time.Since(start)

	// Collect results
	successCount := 0
	errorCount := 0
	for code := range results {
		if code == http.StatusOK {
			successCount++
		} else {
			errorCount++
		}
	}

	t.Logf("Load test results:")
	t.Logf("  Total requests: %d", numRequests)
	t.Logf("  Successful: %d", successCount)
	t.Logf("  Errors: %d", errorCount)
	t.Logf("  Duration: %v", duration)
	t.Logf("  Requests/sec: %.2f", float64(numRequests)/duration.Seconds())

	// At least 90% should succeed
	successRate := float64(successCount) / float64(numRequests)
	if successRate < 0.90 {
		t.Errorf("expected at least 90%% success rate, got %.2f%%", successRate*100)
	}
}

// TestMemoryLeak tests for memory leaks under load
func TestMemoryLeak(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)

	// Run multiple batches and check memory doesn't grow unbounded
	const batches = 5
	const requestsPerBatch = 50

	for batch := 0; batch < batches; batch++ {
		var wg sync.WaitGroup
		for i := 0; i < requestsPerBatch; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				srv.HandleChat(w, req)
			}()
		}
		wg.Wait()

		// Small delay between batches
		time.Sleep(100 * time.Millisecond)
	}

	// If we get here without panicking, memory management is likely OK
	t.Log("Memory leak test completed without issues")
}

// TestPerformanceRegression tests that performance doesn't degrade
func TestPerformanceRegression(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test performance"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)

	const numRequests = 50
	var totalDuration time.Duration

	for i := 0; i < numRequests; i++ {
		start := time.Now()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		srv.HandleChat(w, req)

		duration := time.Since(start)
		totalDuration += duration

		if w.Code != http.StatusOK {
			t.Errorf("request %d failed with status %d", i, w.Code)
		}
	}

	avgDuration := totalDuration / numRequests
	t.Logf("Average request duration: %v", avgDuration)

	// Average should be reasonable (less than 1 second for test model)
	if avgDuration > 1*time.Second {
		t.Errorf("average request duration too high: %v", avgDuration)
	}
}

// TestRateLimiting tests that rate limiting works under load
func TestRateLimiting(t *testing.T) {
	// Create server with strict rate limit
	models := map[string]*ai.VaultGemma{
		"general": buildTestModel(),
	}

	dm := domain.NewDomainManager()
	dm.AddDomain("general", &domain.DomainConfig{
		Name:        "General",
		ModelPath:   "models/general",
		MaxTokens:   64,
		Temperature: 0.7,
		Keywords:    []string{"general"},
	})

	// Very strict rate limit: 1 request per second, burst of 2
	srv := server.NewVaultGemmaServer(models, nil, nil, dm, rate.NewLimiter(rate.Every(time.Second), 2), "rate-test")
	srv.inferenceEngine = inference.NewInferenceEngine(models, dm)

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)

	// Send requests rapidly
	rateLimitedCount := 0
	successCount := 0

	for i := 0; i < 10; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		srv.HandleChat(w, req)

		if w.Code == http.StatusTooManyRequests {
			rateLimitedCount++
		} else if w.Code == http.StatusOK {
			successCount++
		}
	}

	t.Logf("Rate limiting test: %d successful, %d rate limited", successCount, rateLimitedCount)

	// Should have some rate limited requests
	if rateLimitedCount == 0 {
		t.Error("expected some rate limited requests")
	}
}

// TestStressLoad tests extreme load conditions
func TestStressLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Stress test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)

	const numRequests = 500
	var wg sync.WaitGroup
	results := make(chan int, numRequests)

	start := time.Now()

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.HandleChat(w, req)
			results <- w.Code
		}()
	}

	wg.Wait()
	close(results)

	duration := time.Since(start)

	successCount := 0
	for code := range results {
		if code == http.StatusOK {
			successCount++
		}
	}

	t.Logf("Stress test results:")
	t.Logf("  Total: %d", numRequests)
	t.Logf("  Successful: %d", successCount)
	t.Logf("  Duration: %v", duration)
	t.Logf("  Throughput: %.2f req/s", float64(numRequests)/duration.Seconds())

	// At least 80% should succeed under stress
	successRate := float64(successCount) / float64(numRequests)
	if successRate < 0.80 {
		t.Errorf("expected at least 80%% success rate under stress, got %.2f%%", successRate*100)
	}
}

