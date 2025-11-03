package inference

import (
	"context"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

func TestEnhancedInferenceEngine(t *testing.T) {
	// Create a mock model
	model := createMockModel()

	// Create domain manager
	domainManager := domain.NewDomainManager()

	// Create enhanced inference engine
	engine := NewEnhancedInferenceEngine(map[string]*ai.VaultGemma{"test": model}, domainManager)

	// Test request
	req := &EnhancedInferenceRequest{
		Prompt:      "Hello, world!",
		Domain:      "test",
		MaxTokens:   10,
		Temperature: 0.7,
		Model:       model,
		TopP:        0.9,
		TopK:        50,
	}

	// Generate response
	ctx := context.Background()
	response := engine.GenerateEnhancedResponse(ctx, req)

	// Check response
	if response.Error != nil {
		t.Errorf("Expected no error, got: %v", response.Error)
	}

	if response.Content == "" {
		t.Error("Expected non-empty content")
	}

	if response.TokensUsed <= 0 {
		t.Error("Expected positive token count")
	}

	if response.Duration <= 0 {
		t.Error("Expected positive duration")
	}

	t.Logf("Generated response: %s", response.Content)
	t.Logf("Tokens used: %d", response.TokensUsed)
	t.Logf("Duration: %v", response.Duration)
}

func TestEnhancedInferenceEngineWithInvalidModel(t *testing.T) {
	// Create domain manager
	domainManager := domain.NewDomainManager()

	// Create enhanced inference engine with nil model
	engine := NewEnhancedInferenceEngine(map[string]*ai.VaultGemma{}, domainManager)

	// Test request with nil model
	req := &EnhancedInferenceRequest{
		Prompt:      "Hello, world!",
		Domain:      "test",
		MaxTokens:   10,
		Temperature: 0.7,
		Model:       nil,
		TopP:        0.9,
		TopK:        50,
	}

	// Generate response
	ctx := context.Background()
	response := engine.GenerateEnhancedResponse(ctx, req)

	// Check that error is returned
	if response.Error == nil {
		t.Error("Expected error for nil model")
	}
}

func TestEnhancedInferenceEngineWithEmptyPrompt(t *testing.T) {
	// Create a mock model
	model := createMockModel()

	// Create domain manager
	domainManager := domain.NewDomainManager()

	// Create enhanced inference engine
	engine := NewEnhancedInferenceEngine(map[string]*ai.VaultGemma{"test": model}, domainManager)

	// Test request with empty prompt
	req := &EnhancedInferenceRequest{
		Prompt:      "",
		Domain:      "test",
		MaxTokens:   10,
		Temperature: 0.7,
		Model:       model,
		TopP:        0.9,
		TopK:        50,
	}

	// Generate response
	ctx := context.Background()
	response := engine.GenerateEnhancedResponse(ctx, req)

	// Check response
	if response.Error != nil {
		t.Errorf("Expected no error, got: %v", response.Error)
	}

	// Should still generate some response
	if response.Content == "" {
		t.Error("Expected non-empty content even for empty prompt")
	}
}

func TestEnhancedInferenceEngineWithHighTemperature(t *testing.T) {
	// Create a mock model
	model := createMockModel()

	// Create domain manager
	domainManager := domain.NewDomainManager()

	// Create enhanced inference engine
	engine := NewEnhancedInferenceEngine(map[string]*ai.VaultGemma{"test": model}, domainManager)

	// Test request with high temperature
	req := &EnhancedInferenceRequest{
		Prompt:      "Hello, world!",
		Domain:      "test",
		MaxTokens:   10,
		Temperature: 2.0,
		Model:       model,
		TopP:        0.9,
		TopK:        50,
	}

	// Generate response
	ctx := context.Background()
	response := engine.GenerateEnhancedResponse(ctx, req)

	// Check response
	if response.Error != nil {
		t.Errorf("Expected no error, got: %v", response.Error)
	}

	if response.Content == "" {
		t.Error("Expected non-empty content")
	}

	t.Logf("Generated response with high temperature: %s", response.Content)
}

func TestEnhancedInferenceEngineWithLowTopP(t *testing.T) {
	// Create a mock model
	model := createMockModel()

	// Create domain manager
	domainManager := domain.NewDomainManager()

	// Create enhanced inference engine
	engine := NewEnhancedInferenceEngine(map[string]*ai.VaultGemma{"test": model}, domainManager)

	// Test request with low top-p
	req := &EnhancedInferenceRequest{
		Prompt:      "Hello, world!",
		Domain:      "test",
		MaxTokens:   10,
		Temperature: 0.7,
		Model:       model,
		TopP:        0.1,
		TopK:        50,
	}

	// Generate response
	ctx := context.Background()
	response := engine.GenerateEnhancedResponse(ctx, req)

	// Check response
	if response.Error != nil {
		t.Errorf("Expected no error, got: %v", response.Error)
	}

	if response.Content == "" {
		t.Error("Expected non-empty content")
	}

	t.Logf("Generated response with low top-p: %s", response.Content)
}

func TestEnhancedInferenceEngineWithLowTopK(t *testing.T) {
	// Create a mock model
	model := createMockModel()

	// Create domain manager
	domainManager := domain.NewDomainManager()

	// Create enhanced inference engine
	engine := NewEnhancedInferenceEngine(map[string]*ai.VaultGemma{"test": model}, domainManager)

	// Test request with low top-k
	req := &EnhancedInferenceRequest{
		Prompt:      "Hello, world!",
		Domain:      "test",
		MaxTokens:   10,
		Temperature: 0.7,
		Model:       model,
		TopP:        0.9,
		TopK:        5,
	}

	// Generate response
	ctx := context.Background()
	response := engine.GenerateEnhancedResponse(ctx, req)

	// Check response
	if response.Error != nil {
		t.Errorf("Expected no error, got: %v", response.Error)
	}

	if response.Content == "" {
		t.Error("Expected non-empty content")
	}

	t.Logf("Generated response with low top-k: %s", response.Content)
}

func TestEnhancedInferenceEnginePerformance(t *testing.T) {
	// Create a mock model
	model := createMockModel()

	// Create domain manager
	domainManager := domain.NewDomainManager()

	// Create enhanced inference engine
	engine := NewEnhancedInferenceEngine(map[string]*ai.VaultGemma{"test": model}, domainManager)

	// Test request
	req := &EnhancedInferenceRequest{
		Prompt:      "Hello, world!",
		Domain:      "test",
		MaxTokens:   10,
		Temperature: 0.7,
		Model:       model,
		TopP:        0.9,
		TopK:        50,
	}

	// Measure performance
	start := time.Now()
	ctx := context.Background()
	response := engine.GenerateEnhancedResponse(ctx, req)
	duration := time.Since(start)

	// Check response
	if response.Error != nil {
		t.Errorf("Expected no error, got: %v", response.Error)
	}

	// Check performance (should be reasonably fast for a simplified implementation)
	if duration > 60*time.Second {
		t.Errorf("Inference took too long: %v", duration)
	}

	t.Logf("Inference duration: %v", duration)
	t.Logf("Generated response: %s", response.Content)
}

// Helper function to create a mock model for testing
func createMockModel() *ai.VaultGemma {
	config := ai.VaultGemmaConfig{
		HiddenSize:       512,
		NumLayers:        6,
		NumHeads:         8,
		VocabSize:        32000,
		MaxPositionEmbs:  2048,
		IntermediateSize: 2048,
		HeadDim:          64,
		RMSNormEps:       1e-6,
	}

	model := &ai.VaultGemma{
		Config: config,
		Layers: make([]ai.TransformerLayer, config.NumLayers),
		Embed: &ai.EmbeddingLayer{
			Weights: util.NewMatrix64(config.VocabSize, config.HiddenSize),
		},
		Output: &ai.OutputLayer{
			Weights: util.NewMatrix64(config.HiddenSize, config.VocabSize),
		},
	}

	// Initialize embedding weights
	for i := 0; i < config.VocabSize; i++ {
		for j := 0; j < config.HiddenSize; j++ {
			model.Embed.Weights.Data[i*config.HiddenSize+j] = float64(i+j) * 0.01
		}
	}

	// Initialize output weights
	for i := 0; i < config.HiddenSize; i++ {
		for j := 0; j < config.VocabSize; j++ {
			model.Output.Weights.Data[i*config.VocabSize+j] = float64(i+j) * 0.01
		}
	}

	// Initialize layers with mock data
	for i := 0; i < config.NumLayers; i++ {
		model.Layers[i] = ai.TransformerLayer{
			SelfAttention: &ai.MultiHeadAttention{
				NumHeads: config.NumHeads,
				HeadDim:  config.HeadDim,
				WQ:       util.NewMatrix64(config.HiddenSize, config.HiddenSize),
				WK:       util.NewMatrix64(config.HiddenSize, config.HiddenSize),
				WV:       util.NewMatrix64(config.HiddenSize, config.HiddenSize),
				WO:       util.NewMatrix64(config.HiddenSize, config.HiddenSize),
			},
			FeedForward: &ai.FeedForwardNetwork{
				W1: util.NewMatrix64(config.HiddenSize, config.IntermediateSize),
				W2: util.NewMatrix64(config.IntermediateSize, config.HiddenSize),
				W3: util.NewMatrix64(config.HiddenSize, config.IntermediateSize),
			},
			LayerNorm1: &ai.RMSNorm{
				Weight: make([]float64, config.HiddenSize),
				Eps:    config.RMSNormEps,
			},
			LayerNorm2: &ai.RMSNorm{
				Weight: make([]float64, config.HiddenSize),
				Eps:    config.RMSNormEps,
			},
		}

		// Initialize attention weights
		layer := &model.Layers[i]
		for j := 0; j < config.HiddenSize*config.HiddenSize; j++ {
			layer.SelfAttention.WQ.Data[j] = float64(j) * 0.01
			layer.SelfAttention.WK.Data[j] = float64(j) * 0.01
			layer.SelfAttention.WV.Data[j] = float64(j) * 0.01
			layer.SelfAttention.WO.Data[j] = float64(j) * 0.01
		}

		// Initialize feed-forward weights
		for j := 0; j < config.HiddenSize*config.IntermediateSize; j++ {
			layer.FeedForward.W1.Data[j] = float64(j) * 0.01
			layer.FeedForward.W3.Data[j] = float64(j) * 0.01
		}
		for j := 0; j < config.IntermediateSize*config.HiddenSize; j++ {
			layer.FeedForward.W2.Data[j] = float64(j) * 0.01
		}

		// Initialize layer norm weights
		for j := 0; j < config.HiddenSize; j++ {
			layer.LayerNorm1.Weight[j] = 1.0
			layer.LayerNorm2.Weight[j] = 1.0
		}
	}

	return model
}
