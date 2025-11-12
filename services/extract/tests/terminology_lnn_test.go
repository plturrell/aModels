package main

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestAdamOptimizer(t *testing.T) {
	// Test Adam optimizer initialization
	weights := make([][]float32, 2)
	weights[0] = []float32{0.1, 0.2}
	weights[1] = []float32{0.3, 0.4}
	
	optimizer := NewAdamOptimizer(0.001, 0.9, 0.999, 1e-8, weights)
	if optimizer == nil {
		t.Fatal("Adam optimizer should not be nil")
	}
	if optimizer.learningRate != 0.001 {
		t.Errorf("Expected learning rate 0.001, got %f", optimizer.learningRate)
	}
	if optimizer.beta1 != 0.9 {
		t.Errorf("Expected beta1 0.9, got %f", optimizer.beta1)
	}
	if optimizer.beta2 != 0.999 {
		t.Errorf("Expected beta2 0.999, got %f", optimizer.beta2)
	}
	
	// Test optimizer update
	gradients := make([][]float32, 2)
	gradients[0] = []float32{0.01, 0.02}
	gradients[1] = []float32{0.03, 0.04}
	
	initialWeight := weights[0][0]
	optimizer.Update(weights, gradients)
	
	// Weights should have changed
	if weights[0][0] == initialWeight {
		t.Error("Weights should have been updated by Adam optimizer")
	}
}

func TestLiquidLayerTransform(t *testing.T) {
	layer := NewLiquidLayer(10, 5, 10)
	if layer == nil {
		t.Fatal("LiquidLayer should not be nil")
	}
	
	input := make([]float32, 10)
	for i := range input {
		input[i] = float32(i) * 0.1
	}
	
	output := layer.Transform(input, time.Now())
	if len(output) != 10 {
		t.Errorf("Expected output size 10, got %d", len(output))
	}
}

func TestLiquidLayerBatchLearning(t *testing.T) {
	os.Setenv("LNN_BATCH_SIZE", "2")
	defer os.Unsetenv("LNN_BATCH_SIZE")
	
	layer := NewLiquidLayer(10, 5, 10)
	input := make([]float32, 10)
	target := make([]float32, 10)
	
	// First update - should accumulate
	layer.UpdateWeights(input, target, 0.01)
	if layer.batchCount != 1 {
		t.Errorf("Expected batch count 1, got %d", layer.batchCount)
	}
	
	// Second update - should trigger batch update
	layer.UpdateWeights(input, target, 0.01)
	if layer.batchCount != 0 {
		t.Errorf("Expected batch count 0 after batch update, got %d", layer.batchCount)
	}
	
	// Test FlushBatch
	layer.UpdateWeights(input, target, 0.01)
	layer.FlushBatch()
	if layer.batchCount != 0 {
		t.Errorf("Expected batch count 0 after flush, got %d", layer.batchCount)
	}
}

func TestAttentionLayer(t *testing.T) {
	randSource := NewLiquidLayer(10, 5, 10).randSource
	attention := NewAttentionLayer(10, 5, randSource)
	if attention == nil {
		t.Fatal("AttentionLayer should not be nil")
	}
	
	inputs := [][]float32{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
	}
	
	output := attention.Apply(inputs)
	if output == nil {
		t.Fatal("Attention output should not be nil")
	}
	if len(output) != 5 {
		t.Errorf("Expected output size 5, got %d", len(output))
	}
}

func TestSparseVocabulary(t *testing.T) {
	vocab := NewSparseVocabulary(100)
	if vocab == nil {
		t.Fatal("SparseVocabulary should not be nil")
	}
	
	// Test Set and Get
	vocab.Set("term1", 0.5)
	val, ok := vocab.Get("term1")
	if !ok {
		t.Error("Term should exist after Set")
	}
	if val != 0.5 {
		t.Errorf("Expected value 0.5, got %f", val)
	}
	
	// Test Size
	if vocab.Size() != 1 {
		t.Errorf("Expected size 1, got %d", vocab.Size())
	}
	
	// Test pruning
	for i := 0; i < 110; i++ {
		vocab.Set("term"+string(rune(i)), float32(i))
	}
	// Size should be at most maxSize
	if vocab.Size() > 100 {
		t.Errorf("Expected size <= 100 after pruning, got %d", vocab.Size())
	}
}

func TestTerminologyLNNInferDomain(t *testing.T) {
	tnn := NewTerminologyLNN(nil)
	if tnn == nil {
		t.Fatal("TerminologyLNN should not be nil")
	}
	
	ctx := context.Background()
	domain, confidence := tnn.InferDomain(ctx, "customer_id", "customers", nil)
	if domain == "" {
		t.Error("Inferred domain should not be empty")
	}
	if confidence < 0 || confidence > 1 {
		t.Errorf("Confidence should be between 0 and 1, got %f", confidence)
	}
}

func TestTerminologyLNNLearnDomain(t *testing.T) {
	tnn := NewTerminologyLNN(nil)
	ctx := context.Background()
	
	err := tnn.LearnDomain(ctx, "customer_id", "customers", "sales", time.Now())
	if err != nil {
		t.Errorf("LearnDomain should not return error: %v", err)
	}
	
	// Verify domain was learned
	domain, _ := tnn.InferDomain(ctx, "customer_id", "customers", nil)
	if domain == "" {
		t.Error("Domain should be inferred after learning")
	}
}

func TestModelPersistence(t *testing.T) {
	// Create temporary file
	tmpFile := "/tmp/test_lnn_model.json"
	defer os.Remove(tmpFile)
	
	tnn := NewTerminologyLNN(nil)
	ctx := context.Background()
	
	// Learn something
	tnn.LearnDomain(ctx, "test_column", "test_table", "test_domain", time.Now())
	
	// Save model
	err := tnn.SaveModel(tmpFile)
	if err != nil {
		t.Fatalf("SaveModel should not return error: %v", err)
	}
	
	// Verify file exists
	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Fatal("Model file should exist after SaveModel")
	}
	
	// Load model
	tnn2 := NewTerminologyLNN(nil)
	err = tnn2.LoadModel(tmpFile)
	if err != nil {
		t.Fatalf("LoadModel should not return error: %v", err)
	}
	
	// Verify model was loaded
	domain, _ := tnn2.InferDomain(ctx, "test_column", "test_table", nil)
	if domain == "" {
		t.Error("Domain should be inferred after loading model")
	}
}

func TestReproducibility(t *testing.T) {
	os.Setenv("LNN_RANDOM_SEED", "12345")
	defer os.Unsetenv("LNN_RANDOM_SEED")
	
	layer1 := NewLiquidLayer(10, 5, 10)
	layer2 := NewLiquidLayer(10, 5, 10)
	
	// Weights should be identical with same seed
	if len(layer1.weights) != len(layer2.weights) {
		t.Fatal("Layers should have same weight structure")
	}
	
	for i := range layer1.weights {
		for j := range layer1.weights[i] {
			if layer1.weights[i][j] != layer2.weights[i][j] {
				t.Errorf("Weights should be identical with same seed at [%d][%d]", i, j)
			}
		}
	}
}

func BenchmarkLiquidLayerTransform(b *testing.B) {
	layer := NewLiquidLayer(256, 128, 256)
	input := make([]float32, 256)
	for i := range input {
		input[i] = float32(i) * 0.001
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Transform(input, time.Now())
	}
}

func BenchmarkAttentionLayer(b *testing.B) {
	randSource := NewLiquidLayer(256, 128, 256).randSource
	attention := NewAttentionLayer(256, 128, randSource)
	inputs := [][]float32{
		make([]float32, 256),
		make([]float32, 256),
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		attention.Apply(inputs)
	}
}

func BenchmarkTerminologyLNNInferDomain(b *testing.B) {
	tnn := NewTerminologyLNN(nil)
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tnn.InferDomain(ctx, "customer_id", "customers", nil)
	}
}

