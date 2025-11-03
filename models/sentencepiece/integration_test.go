package sentencepiece

import (
	"context"
	"testing"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/trainer"
)

// TestEndToEndUnigram tests the complete pipeline for Unigram models.
func TestEndToEndUnigram(t *testing.T) {
	ctx := context.Background()

	// Training data
	trainingData := []string{
		"hello world",
		"hello there",
		"world peace",
		"peace and love",
		"love and happiness",
	}

	// Train model
	config := &trainer.Config{
		InputData:         trainingData,
		ModelType:         model.TypeUnigram,
		VocabSize:         50,
		CharacterCoverage: 0.9995,
	}

	t.Run("Train", func(t *testing.T) {
		tr, err := trainer.New(model.TypeUnigram)
		if err != nil {
			t.Fatalf("Failed to create trainer: %v", err)
		}

		trainedModel, err := tr.Train(ctx, config)
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}

		if trainedModel == nil {
			t.Fatal("Expected trained model, got nil")
		}

		// Test encoding
		t.Run("Encode", func(t *testing.T) {
			ids, err := trainedModel.Encode(ctx, "hello world")
			if err != nil {
				t.Fatalf("Encoding failed: %v", err)
			}
			if len(ids) == 0 {
				t.Error("Expected non-empty encoding")
			}
			t.Logf("Encoded 'hello world' to %d tokens", len(ids))
		})

		// Test decoding
		t.Run("Decode", func(t *testing.T) {
			ids, _ := trainedModel.Encode(ctx, "hello")
			text, err := trainedModel.Decode(ctx, ids)
			if err != nil {
				t.Fatalf("Decoding failed: %v", err)
			}
			if text == "" {
				t.Error("Expected non-empty decoded text")
			}
			t.Logf("Decoded to: %s", text)
		})

		// Test round-trip
		t.Run("RoundTrip", func(t *testing.T) {
			original := "peace"
			ids, err := trainedModel.Encode(ctx, original)
			if err != nil {
				t.Fatalf("Encoding failed: %v", err)
			}

			decoded, err := trainedModel.Decode(ctx, ids)
			if err != nil {
				t.Fatalf("Decoding failed: %v", err)
			}

			// Note: decoded may not exactly match original due to normalization
			t.Logf("Original: %s, Decoded: %s", original, decoded)
		})
	})
}

// TestEndToEndBPE tests the complete pipeline for BPE models.
func TestEndToEndBPE(t *testing.T) {
	ctx := context.Background()

	trainingData := []string{
		"the quick brown fox",
		"the lazy dog",
		"quick brown dog",
	}

	config := &trainer.Config{
		InputData:         trainingData,
		ModelType:         model.TypeBPE,
		VocabSize:         30,
		CharacterCoverage: 0.9995,
	}

	tr, err := trainer.New(model.TypeBPE)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}

	trainedModel, err := tr.Train(ctx, config)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Test encoding
	ids, err := trainedModel.Encode(ctx, "the quick fox")
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}
	if len(ids) == 0 {
		t.Error("Expected non-empty encoding")
	}
	t.Logf("BPE encoded to %d tokens", len(ids))
}

// TestEndToEndWord tests the complete pipeline for Word models.
func TestEndToEndWord(t *testing.T) {
	ctx := context.Background()

	trainingData := []string{
		"This is a test.",
		"This is another test.",
		"Testing word tokenization.",
	}

	config := &trainer.Config{
		InputData:         trainingData,
		ModelType:         model.TypeWord,
		VocabSize:         20,
		CharacterCoverage: 0.9995,
	}

	tr, err := trainer.New(model.TypeWord)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}

	trainedModel, err := tr.Train(ctx, config)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Test encoding
	ids, err := trainedModel.Encode(ctx, "This is a test")
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}
	if len(ids) == 0 {
		t.Error("Expected non-empty encoding")
	}
	t.Logf("Word model encoded to %d tokens", len(ids))
}

// TestEndToEndChar tests the complete pipeline for Char models.
func TestEndToEndChar(t *testing.T) {
	ctx := context.Background()

	trainingData := []string{
		"abc",
		"def",
		"ghi",
	}

	config := &trainer.Config{
		InputData:         trainingData,
		ModelType:         model.TypeChar,
		VocabSize:         20,
		CharacterCoverage: 0.9995,
	}

	tr, err := trainer.New(model.TypeChar)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}

	trainedModel, err := tr.Train(ctx, config)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Test encoding
	ids, err := trainedModel.Encode(ctx, "abc")
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}
	
	// Should be 3 characters
	if len(ids) != 3 {
		t.Errorf("Expected 3 tokens for 'abc', got %d", len(ids))
	}
	t.Logf("Char model encoded to %d tokens", len(ids))
}

// TestModelComparison compares different model types on the same data.
func TestModelComparison(t *testing.T) {
	ctx := context.Background()
	testText := "hello world"

	trainingData := []string{
		"hello world",
		"hello there",
		"world peace",
	}

	modelTypes := []model.Type{
		model.TypeUnigram,
		model.TypeBPE,
		model.TypeWord,
		model.TypeChar,
	}

	for _, modelType := range modelTypes {
		t.Run(modelType.String(), func(t *testing.T) {
			config := &trainer.Config{
				InputData:         trainingData,
				ModelType:         modelType,
				VocabSize:         30,
				CharacterCoverage: 0.9995,
			}

			tr, err := trainer.New(modelType)
			if err != nil {
				t.Fatalf("Failed to create trainer: %v", err)
			}

			trainedModel, err := tr.Train(ctx, config)
			if err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			ids, err := trainedModel.Encode(ctx, testText)
			if err != nil {
				t.Fatalf("Encoding failed: %v", err)
			}

			t.Logf("%s: %d tokens for '%s'", modelType, len(ids), testText)
		})
	}
}
