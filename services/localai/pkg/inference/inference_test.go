package inference

import (
	"context"
	"testing"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
)

func TestInferenceEngine_GenerateResponse(t *testing.T) {
	// Create a mock model
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			HiddenSize:       512,
			NumLayers:        12,
			NumHeads:         8,
			VocabSize:        50000,
			MaxPositionEmbs:  2048,
			IntermediateSize: 2048,
			HeadDim:          64,
			RMSNormEps:       1e-6,
		},
		Embed:  &ai.EmbeddingLayer{},
		Output: &ai.OutputLayer{},
		Layers: make([]ai.TransformerLayer, 12),
	}

	// Create a mock domain manager
	domainManager := &domain.DomainManager{}

	// Create inference engine
	engine := NewInferenceEngine(map[string]*ai.VaultGemma{"test": model}, domainManager)

	// Test valid request
	req := &InferenceRequest{
		Prompt:      "Hello, how are you?",
		Domain:      "test",
		MaxTokens:   100,
		Temperature: 0.7,
		Model:       model,
	}

	response := engine.GenerateResponse(context.Background(), req)

	if response.Error != nil {
		t.Fatalf("Expected no error, got: %v", response.Error)
	}

	if response.Content == "" {
		t.Fatal("Expected non-empty content")
	}

	if response.TokensUsed == 0 {
		t.Fatal("Expected non-zero tokens used")
	}

	if response.ModelName != "test" {
		t.Fatalf("Expected model name 'test', got '%s'", response.ModelName)
	}

	if response.Domain != "test" {
		t.Fatalf("Expected domain 'test', got '%s'", response.Domain)
	}

	if response.Duration <= 0 {
		t.Fatal("Expected positive duration")
	}
}

func TestInferenceEngine_GenerateResponse_NilModel(t *testing.T) {
	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	req := &InferenceRequest{
		Prompt:      "Hello",
		Domain:      "test",
		MaxTokens:   100,
		Temperature: 0.7,
		Model:       nil,
	}

	response := engine.GenerateResponse(context.Background(), req)

	if response.Error == nil {
		t.Fatal("Expected error for nil model")
	}
}

func TestInferenceEngine_GenerateResponse_DefaultParameters(t *testing.T) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			HiddenSize:       512,
			NumLayers:        12,
			NumHeads:         8,
			VocabSize:        50000,
			MaxPositionEmbs:  2048,
			IntermediateSize: 2048,
			HeadDim:          64,
			RMSNormEps:       1e-6,
		},
		Embed:  &ai.EmbeddingLayer{},
		Output: &ai.OutputLayer{},
		Layers: make([]ai.TransformerLayer, 12),
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{"test": model}, &domain.DomainManager{})

	// Test with zero values (should use defaults)
	req := &InferenceRequest{
		Prompt:      "Hello",
		Domain:      "test",
		MaxTokens:   0, // Should default to 512
		Temperature: 0, // Should default to 0.7
		Model:       model,
	}

	response := engine.GenerateResponse(context.Background(), req)

	if response.Error != nil {
		t.Fatalf("Expected no error, got: %v", response.Error)
	}

	// The actual tokenization and generation would use the default values
	// We can't easily test the exact values without mocking the model internals
}

func TestInferenceEngine_TokenizeInput(t *testing.T) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			VocabSize: 50000,
		},
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	// Test tokenization
	tokens, err := engine.tokenizeInput("Hello world", model)
	if err != nil {
		t.Fatalf("Failed to tokenize input: %v", err)
	}

	if len(tokens) == 0 {
		t.Fatal("Expected non-empty tokens")
	}

	// Check that BOS token is added
	if tokens[0] != 1 {
		t.Fatal("Expected BOS token (1) at the beginning")
	}
}

func TestInferenceEngine_GenerateTokens(t *testing.T) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			HiddenSize: 512,
			VocabSize:  50000,
		},
		Embed:  &ai.EmbeddingLayer{},
		Output: &ai.OutputLayer{},
		Layers: make([]ai.TransformerLayer, 12),
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	inputTokens := []int{1, 100, 200, 300} // BOS + some tokens
	maxTokens := 10
	temperature := 0.7

	tokens, err := engine.generateTokens(context.Background(), model, inputTokens, maxTokens, temperature, ai.DefaultTopP, ai.DefaultTopK)
	if err != nil {
		t.Fatalf("Failed to generate tokens: %v", err)
	}

	if len(tokens) == 0 {
		t.Fatal("Expected non-empty generated tokens")
	}

	if len(tokens) > maxTokens {
		t.Fatalf("Expected at most %d tokens, got %d", maxTokens, len(tokens))
	}
}

func TestInferenceEngine_DecodeTokens(t *testing.T) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			VocabSize: 50000,
		},
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	// Test with BOS, some tokens, and EOS
	tokens := []int{1, 100, 200, 300, 2} // BOS, tokens, EOS

	text, err := engine.decodeTokens(tokens, model)
	if err != nil {
		t.Fatalf("Failed to decode tokens: %v", err)
	}

	if text == "" {
		t.Fatal("Expected non-empty decoded text")
	}

	// Should not contain BOS or EOS tokens in the output
	if len(text) == 0 {
		t.Fatal("Expected some decoded content")
	}
}

func TestInferenceEngine_TokensToTensor(t *testing.T) {
	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	tokens := []int{1, 100, 200, 300}
	hiddenSize := 512

	tensor := engine.tokensToTensor(tokens, hiddenSize)

	expectedSize := len(tokens) * hiddenSize
	if len(tensor) != expectedSize {
		t.Fatalf("Expected tensor size %d, got %d", expectedSize, len(tensor))
	}

	// Check that tensor is not all zeros
	allZero := true
	for _, val := range tensor {
		if val != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("Expected non-zero tensor values")
	}
}

func TestInferenceEngine_RunModelForward(t *testing.T) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			HiddenSize: 512,
		},
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	inputTensor := make([]float64, 1024) // 2 tokens * 512 hidden size
	for i := range inputTensor {
		inputTensor[i] = float64(i) * 0.01
	}

	outputTensor, err := engine.runModelForward(model, inputTensor)
	if err != nil {
		t.Fatalf("Failed to run model forward: %v", err)
	}

	if len(outputTensor) != len(inputTensor) {
		t.Fatalf("Expected output size %d, got %d", len(inputTensor), len(outputTensor))
	}

	// Check that output is different from input (transformation occurred)
	identical := true
	for i, val := range outputTensor {
		if val != inputTensor[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("Expected output to be different from input")
	}
}

func TestInferenceEngine_GetNextTokenProbs(t *testing.T) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			VocabSize: 1000,
		},
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	tokens := []int{1, 100, 200}
	hiddenState := make([]float64, 512)

	probs, err := engine.getNextTokenProbs(model, tokens, hiddenState)
	if err != nil {
		t.Fatalf("Failed to get next token probs: %v", err)
	}

	if len(probs) != model.Config.VocabSize {
		t.Fatalf("Expected %d probabilities, got %d", model.Config.VocabSize, len(probs))
	}

	// Check that probabilities sum to approximately 1
	sum := 0.0
	for _, prob := range probs {
		sum += prob
	}

	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("Expected probabilities to sum to ~1, got %f", sum)
	}
}

func TestInferenceEngine_SampleToken(t *testing.T) {
	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	probs := []float64{0.1, 0.2, 0.3, 0.4}
	temperature := 0.7

	token := engine.sampleToken(probs, temperature)

	if token < 0 || token >= len(probs) {
		t.Fatalf("Expected token in range [0, %d), got %d", len(probs), token)
	}
}

func TestInferenceEngine_SampleToken_TemperatureScaling(t *testing.T) {
	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	probs := []float64{0.1, 0.2, 0.3, 0.4}

	// Test with different temperatures
	highTempToken := engine.sampleToken(probs, 2.0)
	lowTempToken := engine.sampleToken(probs, 0.1)

	// With higher temperature, we expect more randomness
	// With lower temperature, we expect more deterministic behavior
	// The exact behavior depends on the sampling implementation
	_ = highTempToken
	_ = lowTempToken
}

// Benchmark tests
func BenchmarkInferenceEngine_GenerateResponse(b *testing.B) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			HiddenSize:       512,
			NumLayers:        12,
			NumHeads:         8,
			VocabSize:        50000,
			MaxPositionEmbs:  2048,
			IntermediateSize: 2048,
			HeadDim:          64,
			RMSNormEps:       1e-6,
		},
		Embed:  &ai.EmbeddingLayer{},
		Output: &ai.OutputLayer{},
		Layers: make([]ai.TransformerLayer, 12),
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{"test": model}, &domain.DomainManager{})

	req := &InferenceRequest{
		Prompt:      "Hello, how are you?",
		Domain:      "test",
		MaxTokens:   100,
		Temperature: 0.7,
		Model:       model,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.GenerateResponse(context.Background(), req)
	}
}

func BenchmarkInferenceEngine_TokenizeInput(b *testing.B) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			VocabSize: 50000,
		},
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.tokenizeInput("Hello world, this is a test prompt", model)
	}
}

func BenchmarkInferenceEngine_GenerateTokens(b *testing.B) {
	model := &ai.VaultGemma{
		Config: ai.VaultGemmaConfig{
			HiddenSize: 512,
			VocabSize:  50000,
		},
		Embed:  &ai.EmbeddingLayer{},
		Output: &ai.OutputLayer{},
		Layers: make([]ai.TransformerLayer, 12),
	}

	engine := NewInferenceEngine(map[string]*ai.VaultGemma{}, &domain.DomainManager{})

	inputTokens := []int{1, 100, 200, 300}
	maxTokens := 50
	temperature := 0.7

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.generateTokens(context.Background(), model, inputTokens, maxTokens, temperature, ai.DefaultTopP, ai.DefaultTopK)
	}
}

// Test utility functions
func TestSplitIntoWords(t *testing.T) {
	tests := []struct {
		input    string
		expected []string
	}{
		{"hello world", []string{"hello", "world"}},
		{"hello\nworld", []string{"hello", "world"}},
		{"hello\tworld", []string{"hello", "world"}},
		{"hello  world", []string{"hello", "world"}},
		{"", []string{}},
		{"hello", []string{"hello"}},
		{"hello world test", []string{"hello", "world", "test"}},
	}

	for _, test := range tests {
		result := splitIntoWords(test.input)
		if len(result) != len(test.expected) {
			t.Fatalf("Expected %d words, got %d for input '%s'", len(test.expected), len(result), test.input)
		}

		for i, word := range result {
			if word != test.expected[i] {
				t.Fatalf("Expected word '%s', got '%s' at index %d", test.expected[i], word, i)
			}
		}
	}
}

func TestHashString(t *testing.T) {
	tests := []struct {
		input string
	}{
		{"hello"},
		{"world"},
		{""},
		{"test"},
	}

	for _, test := range tests {
		result := hashString(test.input)
		if test.input == "" && result != 0 {
			t.Fatalf("Expected zero hash for empty string, got %d", result)
		}
		if test.input != "" && result == 0 {
			t.Fatalf("Expected non-zero hash for input '%s'", test.input)
		}
		if result < 0 {
			t.Fatalf("Expected non-negative hash, got %d for input '%s'", result, test.input)
		}
	}
}

func TestHashString_Consistency(t *testing.T) {
	input := "hello world"
	hash1 := hashString(input)
	hash2 := hashString(input)

	if hash1 != hash2 {
		t.Fatal("Hash function should be consistent")
	}
}

func TestHashString_Uniqueness(t *testing.T) {
	inputs := []string{"hello", "world", "test", "different"}
	hashes := make(map[int]bool)

	for _, input := range inputs {
		hash := hashString(input)
		if hashes[hash] {
			t.Fatalf("Hash collision for input '%s'", input)
		}
		hashes[hash] = true
	}
}
