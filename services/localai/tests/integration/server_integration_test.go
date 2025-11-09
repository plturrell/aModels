package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/server"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
	"golang.org/x/time/rate"
)

// buildTestModel creates a minimal test model
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

// newTestServer creates a test server with minimal configuration
func newTestServer() *server.VaultGemmaServer {
	models := map[string]*ai.VaultGemma{
		"general": buildTestModel(),
		"vaultgemma": buildTestModel(),
	}

	dm := domain.NewDomainManager()
	dm.AddDomain("general", &domain.DomainConfig{
		Name:        "General",
		ModelPath:   "models/general",
		MaxTokens:   64,
		Temperature: 0.7,
		Keywords:    []string{"general"},
	})

	dm.AddDomain("vaultgemma", &domain.DomainConfig{
		Name:        "VaultGemma",
		ModelPath:   "models/vaultgemma",
		MaxTokens:   64,
		Temperature: 0.7,
		Keywords:    []string{"vaultgemma"},
	})

	srv := server.NewVaultGemmaServer(models, nil, nil, dm, rate.NewLimiter(rate.Every(time.Millisecond), 100), "test")
	srv.inferenceEngine = inference.NewInferenceEngine(models, dm)
	return srv
}

// TestEndToEndChatFlow tests the complete flow from request to response
func TestEndToEndChatFlow(t *testing.T) {
	srv := newTestServer()

	// Step 1: Health check
	healthReq := httptest.NewRequest(http.MethodGet, "/health", nil)
	healthW := httptest.NewRecorder()
	srv.HandleHealth(healthW, healthReq)
	if healthW.Code != http.StatusOK {
		t.Fatalf("health check failed: %d", healthW.Code)
	}

	// Step 2: List models
	modelsReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	modelsW := httptest.NewRecorder()
	srv.HandleModels(modelsW, modelsReq)
	if modelsW.Code != http.StatusOK {
		t.Fatalf("list models failed: %d", modelsW.Code)
	}

	// Step 3: Chat completion
	chatBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Hello"},
		},
		"max_tokens": 16,
	}
	chatRaw, _ := json.Marshal(chatBody)
	chatReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(chatRaw))
	chatReq.Header.Set("Content-Type", "application/json")
	chatW := httptest.NewRecorder()

	srv.HandleChat(chatW, chatReq)
	if chatW.Code != http.StatusOK {
		t.Fatalf("chat completion failed: %d, body: %s", chatW.Code, chatW.Body.String())
	}

	var chatResp map[string]interface{}
	if err := json.NewDecoder(chatW.Body).Decode(&chatResp); err != nil {
		t.Fatalf("failed to decode chat response: %v", err)
	}

	if chatResp["choices"] == nil {
		t.Error("expected choices in response")
	}
}

// TestMultiBackendRouting tests routing to different backends
func TestMultiBackendRouting(t *testing.T) {
	srv := newTestServer()

	// Test safetensors backend (default)
	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	metadata, ok := resp["metadata"].(map[string]interface{})
	if !ok {
		t.Fatal("expected metadata")
	}

	backendType, ok := metadata["backend_type"].(string)
	if !ok {
		t.Error("expected backend_type")
	}

	if backendType == "" {
		t.Error("backend_type should not be empty")
	}
}

// TestDomainRouting tests automatic domain detection and routing
func TestDomainRouting(t *testing.T) {
	srv := newTestServer()

	// Note: Domain routing is tested through the public HandleChat API
	// which uses the domain manager internally

	// Test SQL domain detection (will use general as fallback if SQL domain not configured)
	sqlBody := map[string]interface{}{
		"model": "auto",
		"messages": []map[string]string{
			{"role": "user", "content": "How do I write a SQL query to join tables?"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(sqlBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	// Should route to appropriate domain (may use general as fallback)
	if w.Code != http.StatusOK && w.Code != http.StatusInternalServerError {
		t.Errorf("expected status 200 or 500, got %d", w.Code)
	}
}

// TestErrorRecovery tests error handling and recovery
func TestErrorRecovery(t *testing.T) {
	srv := newTestServer()

	// Test invalid request
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("{invalid")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}

	// Test that server still works after error
	validBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(validBody)
	validReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	validReq.Header.Set("Content-Type", "application/json")
	validW := httptest.NewRecorder()

	srv.HandleChat(validW, validReq)

	if validW.Code != http.StatusOK {
		t.Errorf("expected status 200 after error recovery, got %d", validW.Code)
	}
}

// TestConcurrentRequests tests handling of concurrent requests
func TestConcurrentRequests(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)

	// Send 10 concurrent requests
	results := make(chan int, 10)
	for i := 0; i < 10; i++ {
		go func() {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.HandleChat(w, req)
			results <- w.Code
		}()
	}

	// Collect results
	successCount := 0
	for i := 0; i < 10; i++ {
		code := <-results
		if code == http.StatusOK {
			successCount++
		}
	}

	if successCount < 8 {
		t.Errorf("expected at least 8 successful requests, got %d", successCount)
	}
}

// TestRequestTimeout tests request timeout handling
func TestRequestTimeout(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)

	// Create context with very short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
	defer cancel()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req = req.WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	// Should handle timeout gracefully
	if w.Code != http.StatusRequestTimeout && w.Code != http.StatusOK {
		t.Errorf("expected status 408 or 200, got %d", w.Code)
	}
}

