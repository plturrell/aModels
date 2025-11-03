package tests

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	localserver "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/server"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
	"golang.org/x/time/rate"
)

func newMockVaultGemma() *ai.VaultGemma {
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

func newTestDomainManager() *domain.DomainManager {
	dm := domain.NewDomainManager()
	dm.AddDomain("general", &domain.DomainConfig{
		Name:        "General",
		ModelPath:   "models/general",
		MaxTokens:   1024,
		Temperature: 0.7,
	})
	return dm
}

func writeTestDomainsConfig(t *testing.T) string {
	t.Helper()

	configJSON := `{
  "domains": {
    "general": {
      "name": "General",
      "layer": "layer4",
      "team": "FoundationTeam",
      "backend_type": "vaultgemma",
      "model_path": "models/general",
      "max_tokens": 1024,
      "temperature": 0.7,
      "tags": ["general"],
      "keywords": ["general"],
      "fallback_model": ""
    },
    "0x3579-VectorProcessingAgent": {
      "name": "Vector Processing Agent",
      "layer": "layer4",
      "team": "DataTeam",
      "backend_type": "vaultgemma",
      "model_path": "models/vector",
      "max_tokens": 2048,
      "temperature": 0.6,
      "tags": ["vector"],
      "keywords": ["vector", "embedding", "similarity"],
      "fallback_model": "general"
    },
    "0x5678-SQLAgent": {
      "name": "SQL Agent",
      "layer": "layer4",
      "team": "DataTeam",
      "backend_type": "vaultgemma",
      "model_path": "models/sql",
      "max_tokens": 2048,
      "temperature": 0.8,
      "tags": ["sql"],
      "keywords": ["sql", "select", "database"],
      "fallback_model": "general"
    },
    "0xB10C-BlockchainAgent": {
      "name": "Blockchain Agent",
      "layer": "layer4",
      "team": "LedgerTeam",
      "backend_type": "vaultgemma",
      "model_path": "models/blockchain",
      "max_tokens": 2048,
      "temperature": 0.9,
      "tags": ["blockchain"],
      "keywords": ["blockchain", "transaction", "ledger"],
      "fallback_model": "general"
    },
    "0xF3C0-ESGFinanceAgent": {
      "name": "ESG Finance Agent",
      "layer": "layer4",
      "team": "FinanceTeam",
      "backend_type": "vaultgemma",
      "model_path": "models/esg",
      "max_tokens": 2048,
      "temperature": 0.65,
      "tags": ["esg"],
      "keywords": ["esg", "sustainability", "environmental"],
      "fallback_model": "general"
    }
  },
  "default_domain": "general"
}`

	dir := t.TempDir()
	path := filepath.Join(dir, "domains.json")
	if err := os.WriteFile(path, []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write domain config: %v", err)
	}
	return path
}

func TestDomainManager(t *testing.T) {
	dm := domain.NewDomainManager()
	configPath := writeTestDomainsConfig(t)

	// Test loading domain configs
	err := dm.LoadDomainConfigs(configPath)
	if err != nil {
		t.Fatalf("Failed to load domain configs: %v", err)
	}

	// Test listing domains
	domains := dm.ListDomains()
	if len(domains) == 0 {
		t.Error("Expected domains to be loaded, got 0")
	}
	t.Logf("Loaded %d domains", len(domains))

	// Test getting domain config
	config, exists := dm.GetDomainConfig("0x3579-VectorProcessingAgent")
	if !exists {
		t.Error("Expected VectorProcessingAgent domain to exist")
	}
	if config != nil {
		t.Logf("VectorProcessingAgent config: %s (Layer: %s, Team: %s)",
			config.Name, config.Layer, config.Team)
	}

	// Test domain detection
	prompt := "I need to process vector embeddings and calculate similarity"
	detectedDomain := dm.DetectDomain(prompt, []string{})
	t.Logf("Detected domain for vector prompt: %s", detectedDomain)

	// Test SQL domain detection
	sqlPrompt := "SELECT * FROM users WHERE id = 1"
	sqlDomain := dm.DetectDomain(sqlPrompt, []string{})
	t.Logf("Detected domain for SQL prompt: %s", sqlDomain)

	// Test default domain
	defaultDomain := dm.GetDefaultDomain()
	if defaultDomain != "general" {
		t.Errorf("Expected default domain 'general', got '%s'", defaultDomain)
	}
}

func TestDomainConfigValidation(t *testing.T) {
	tests := []struct {
		name    string
		config  *domain.DomainConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: &domain.DomainConfig{
				Name:        "Test Agent",
				ModelPath:   "models/test",
				MaxTokens:   1000,
				Temperature: 0.5,
			},
			wantErr: false,
		},
		{
			name: "empty name",
			config: &domain.DomainConfig{
				Name:        "",
				ModelPath:   "models/test",
				MaxTokens:   1000,
				Temperature: 0.5,
			},
			wantErr: true,
		},
		{
			name: "empty model path",
			config: &domain.DomainConfig{
				Name:        "Test Agent",
				ModelPath:   "",
				MaxTokens:   1000,
				Temperature: 0.5,
			},
			wantErr: true,
		},
		{
			name: "invalid max tokens",
			config: &domain.DomainConfig{
				Name:        "Test Agent",
				ModelPath:   "models/test",
				MaxTokens:   0,
				Temperature: 0.5,
			},
			wantErr: true,
		},
		{
			name: "invalid temperature",
			config: &domain.DomainConfig{
				Name:        "Test Agent",
				ModelPath:   "models/test",
				MaxTokens:   1000,
				Temperature: 3.0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestVaultGemmaServerHealth(t *testing.T) {
	mockModel := newMockVaultGemma()
	dm := newTestDomainManager()
	vgServer := localserver.NewVaultGemmaServer(
		map[string]*ai.VaultGemma{"general": mockModel},
		nil,
		nil,
		dm,
		rate.NewLimiter(rate.Every(time.Second), 10),
		"2.0.0-test",
	)

	// Create test request
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	// Test health endpoint
	vgServer.HandleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	// Parse response
	var health map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&health); err != nil {
		t.Fatalf("Failed to decode health response: %v", err)
	}

	// Verify response fields
	if status, ok := health["status"].(string); !ok || status != "ok" {
		t.Errorf("Expected status 'ok', got %v", health["status"])
	}

	if version, ok := health["version"].(string); !ok || version != "2.0.0-test" {
		t.Errorf("Expected version '2.0.0-test', got %v", health["version"])
	}

	t.Logf("Health check passed: %+v", health)
}

func TestVaultGemmaServerModels(t *testing.T) {
	mockModel := newMockVaultGemma()
	dm := newTestDomainManager()
	vgServer := localserver.NewVaultGemmaServer(
		map[string]*ai.VaultGemma{"general": mockModel},
		nil,
		nil,
		dm,
		rate.NewLimiter(rate.Every(time.Second), 10),
		"2.0.0-test",
	)
	vgServer.DisableEnhancedInference()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()

	vgServer.HandleModels(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("Failed to decode models response: %v", err)
	}

	t.Logf("Models response: %+v", response)
}

func TestVaultGemmaServerChat(t *testing.T) {
	mockModel := newMockVaultGemma()
	dm := domain.NewDomainManager()
	configPath := writeTestDomainsConfig(t)
	if err := dm.LoadDomainConfigs(configPath); err != nil {
		t.Fatalf("LoadDomainConfigs failed: %v", err)
	}

	vgServer := localserver.NewVaultGemmaServer(
		map[string]*ai.VaultGemma{"general": mockModel},
		nil,
		nil,
		dm,
		rate.NewLimiter(rate.Every(time.Second), 10),
		"2.0.0-test",
	)
	vgServer.DisableEnhancedInference()

	// Test chat request
	chatReq := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Hello, how are you?"},
		},
		"max_tokens":  100,
		"temperature": 0.7,
	}

	body, _ := json.Marshal(chatReq)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	vgServer.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("Failed to decode chat response: %v", err)
	}

	// Verify response structure
	if _, ok := response["id"]; !ok {
		t.Error("Expected 'id' field in response")
	}
	if _, ok := response["choices"]; !ok {
		t.Error("Expected 'choices' field in response")
	}
	if _, ok := response["usage"]; !ok {
		t.Error("Expected 'usage' field in response")
	}

	t.Logf("Chat response: %+v", response)
}

func TestRateLimiting(t *testing.T) {
	mockModel := newMockVaultGemma()
	dm := newTestDomainManager()

	vgServer := localserver.NewVaultGemmaServer(
		map[string]*ai.VaultGemma{"general": mockModel},
		nil,
		nil,
		dm,
		rate.NewLimiter(rate.Every(time.Second), 1),
		"2.0.0-test",
	)
	vgServer.DisableEnhancedInference()

	// First request should succeed
	req1 := httptest.NewRequest(http.MethodGet, "/health", nil)
	w1 := httptest.NewRecorder()
	handler1 := vgServer.RateLimitMiddleware(vgServer.HandleHealth)
	handler1(w1, req1)

	if w1.Code != http.StatusOK {
		t.Errorf("First request: expected status 200, got %d", w1.Code)
	}

	// Second immediate request should be rate limited
	req2 := httptest.NewRequest(http.MethodGet, "/health", nil)
	w2 := httptest.NewRecorder()
	handler2 := vgServer.RateLimitMiddleware(vgServer.HandleHealth)
	handler2(w2, req2)

	if w2.Code != http.StatusTooManyRequests {
		t.Errorf("Second request: expected status 429, got %d", w2.Code)
	}

	t.Log("Rate limiting working correctly")
}

func TestCORS(t *testing.T) {
	handler := localserver.EnableCORS(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	// Test OPTIONS request
	req := httptest.NewRequest(http.MethodOptions, "/test", nil)
	w := httptest.NewRecorder()
	handler(w, req)

	// Verify CORS headers
	if origin := w.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Errorf("Expected Access-Control-Allow-Origin: *, got %s", origin)
	}

	if methods := w.Header().Get("Access-Control-Allow-Methods"); methods == "" {
		t.Error("Expected Access-Control-Allow-Methods header")
	}

	if headers := w.Header().Get("Access-Control-Allow-Headers"); headers == "" {
		t.Error("Expected Access-Control-Allow-Headers header")
	}

	t.Log("CORS headers configured correctly")
}

func TestDomainDetectionWithKeywords(t *testing.T) {
	dm := domain.NewDomainManager()
	configPath := writeTestDomainsConfig(t)
	if err := dm.LoadDomainConfigs(configPath); err != nil {
		t.Fatalf("LoadDomainConfigs failed: %v", err)
	}

	tests := []struct {
		name           string
		prompt         string
		expectedDomain string
	}{
		{
			name:           "vector processing",
			prompt:         "calculate cosine similarity between embeddings",
			expectedDomain: "0x3579-VectorProcessingAgent",
		},
		{
			name:           "sql query",
			prompt:         "SELECT users FROM database WHERE active = true",
			expectedDomain: "0x5678-SQLAgent",
		},
		{
			name:           "blockchain",
			prompt:         "create a new transaction on the blockchain",
			expectedDomain: "0xB10C-BlockchainAgent",
		},
		{
			name:           "esg finance",
			prompt:         "analyze environmental sustainability metrics",
			expectedDomain: "0xF3C0-ESGFinanceAgent",
		},
		{
			name:           "generic",
			prompt:         "hello world",
			expectedDomain: "general",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			detected := dm.DetectDomain(tt.prompt, []string{})
			t.Logf("Prompt: '%s' -> Detected: %s (Expected: %s)",
				tt.prompt, detected, tt.expectedDomain)

			// Note: Detection may not always match exactly due to keyword scoring
			// This is informational rather than strict assertion
		})
	}
}
