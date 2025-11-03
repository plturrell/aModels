package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
	"golang.org/x/time/rate"
)

func newTestServer() *VaultGemmaServer {
	models := map[string]*ai.VaultGemma{
		"general": buildCompactModel(),
	}

	dm := domain.NewDomainManager()
	dm.AddDomain("general", &domain.DomainConfig{
		Name:        "General",
		ModelPath:   "models/general",
		MaxTokens:   64,
		Temperature: 0.7,
		DomainTags:  []string{"general"},
		Keywords:    []string{"general"},
	})

	srv := NewVaultGemmaServer(models, nil, nil, dm, rate.NewLimiter(rate.Every(time.Millisecond), 100), "1.0-test")
	srv.inferenceEngine = inference.NewInferenceEngine(models, dm)
	srv.enhancedEngine = nil
	srv.enhancedLogging = NewEnhancedLogging(&MockHANALogger{})
	return srv
}

func TestHandleModels(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()

	srv.HandleModels(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp["object"] != "list" {
		t.Fatalf("expected object 'list', got %v", resp["object"])
	}
}

func TestHandleDomainRegistry(t *testing.T) {
	srv := newTestServer()
	srv.UpdateAgentCatalog(AgentCatalog{Suites: []AgentSuite{{Name: "agentic", ToolNames: []string{"search_documents"}}}})

	req := httptest.NewRequest(http.MethodGet, "/v1/domain-registry", nil)
	w := httptest.NewRecorder()

	srv.HandleDomainRegistry(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp ModelRegistry
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode domain registry: %v", err)
	}

	if _, ok := resp.Domains["general"]; !ok {
		t.Fatalf("expected 'general' domain in registry")
	}
	if resp.AgentCatalog == nil {
		t.Fatalf("expected agent catalog in registry response")
	}
	if len(resp.AgentCatalog.Suites) == 0 {
		t.Fatalf("expected suites in agent catalog")
	}
	if resp.AgentCatalogUpdated == "" {
		t.Fatalf("expected catalog updated timestamp")
	}
}

func TestEnrichPromptWithAgentCatalog(t *testing.T) {
	srv := newTestServer()
	srv.UpdateAgentCatalog(AgentCatalog{Suites: []AgentSuite{{Name: "agentic", ToolNames: []string{"search_documents"}}}, Tools: []AgentTool{{Name: "search_documents", Description: "Search docs"}}})

	prompt := srv.enrichPromptWithAgentCatalog("Investigate logs")

	if !strings.Contains(prompt, "Suite agentic") {
		t.Fatalf("expected catalog context in prompt, got %q", prompt)
	}
}

func TestHandleHealth(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	srv.HandleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if w.Body.Len() == 0 {
		t.Fatalf("expected health response body, got empty")
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode health response: %v", err)
	}

	if resp["status"] != "ok" {
		t.Fatalf("expected status 'ok', got %v", resp["status"])
	}
}

func TestHandleChatWithFallback(t *testing.T) {
	srv := newTestServer()

	// Add fallback-only domain to domain manager without model entry
	srv.domainManager.AddDomain("sql", &domain.DomainConfig{
		Name:          "SQL",
		ModelPath:     "models/sql",
		MaxTokens:     64,
		Temperature:   0.6,
		FallbackModel: "general",
		Keywords:      []string{"sql"},
	})

	reqBody := map[string]interface{}{
		"model": "sql",
		"messages": []map[string]string{
			{"role": "user", "content": "Explain SQL joins"},
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

	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode chat response: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatalf("expected at least one choice in response")
	}

	content := resp.Choices[0].Message.Content
	if resp.Metadata == nil {
		t.Fatalf("expected metadata in response")
	}

	fallbackUsed, ok := resp.Metadata["fallback_used"].(bool)
	if !ok || !fallbackUsed {
		t.Fatalf("expected fallback_used=true, got %v", resp.Metadata["fallback_used"])
	}

	modelKey, ok := resp.Metadata["fallback_model"].(string)
	if !ok || modelKey != "general" {
		t.Fatalf("expected fallback_model 'general', got %v", resp.Metadata["fallback_model"])
	}

	if !strings.Contains(content, "token_") {
		t.Fatalf("expected generated content tokens, got %q", content)
	}
}

func buildCompactModel() *ai.VaultGemma {
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

func TestRateLimitMiddleware(t *testing.T) {
	srv := newTestServer()
	srv.limiter = rate.NewLimiter(rate.Every(time.Hour), 0) // always deny

	handler := srv.RateLimitMiddleware(func(http.ResponseWriter, *http.Request) {})

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	handler(w, req)

	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429 status from rate limit middleware, got %d", w.Code)
	}
}
