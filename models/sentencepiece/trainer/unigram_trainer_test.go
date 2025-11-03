package trainer

import (
	"context"
	"testing"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

func TestUnigramTrainer_Basic(t *testing.T) {
	config := &Config{
		InputData: []string{
			"hello world",
			"hello there",
			"world peace",
		},
		ModelType:         model.TypeUnigram,
		VocabSize:         100,
		CharacterCoverage: 0.9995,
		MaxSentenceLength: 1000,
	}

	trainer, err := New(model.TypeUnigram)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}

	trainedModel, err := trainer.Train(context.Background(), config)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	
	if trainedModel == nil {
		t.Error("Expected trained model, got nil")
	}
	
	// Verify model can encode
	ctx := context.Background()
	ids, err := trainedModel.Encode(ctx, "hello")
	if err != nil {
		t.Errorf("Failed to encode with trained model: %v", err)
	}
	if len(ids) == 0 {
		t.Error("Expected non-empty encoding")
	}
}

func TestUnigramTrainer_EmptyData(t *testing.T) {
	config := &Config{
		InputData:         []string{},
		ModelType:         model.TypeUnigram,
		VocabSize:         100,
		CharacterCoverage: 0.9995,
	}

	trainer, err := New(model.TypeUnigram)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}

	_, err = trainer.Train(context.Background(), config)
	if err == nil {
		t.Error("Expected error for empty training data")
	}
}

func TestUnigramTrainer_LargeSentences(t *testing.T) {
	// Create a sentence that exceeds max length
	longSentence := ""
	for i := 0; i < 1000; i++ {
		longSentence += "a"
	}

	config := &Config{
		InputData: []string{
			"short",
			longSentence,
		},
		ModelType:         model.TypeUnigram,
		VocabSize:         50,
		MaxSentenceLength: 100, // Should filter out long sentence
	}

	trainer, err := New(model.TypeUnigram)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}

	trainedModel, err := trainer.Train(context.Background(), config)
	// Should process successfully (filtering long sentences)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	if trainedModel == nil {
		t.Error("Expected trained model, got nil")
	}
}

func TestNew_UnsupportedModelType(t *testing.T) {
	_, err := New(model.Type(999))
	if err == nil {
		t.Error("Expected error for unsupported model type")
	}
}

func TestNew_AllModelTypes(t *testing.T) {
	types := []model.Type{
		model.TypeUnigram,
		model.TypeBPE,
		model.TypeWord,
		model.TypeChar,
	}

	for _, modelType := range types {
		trainer, err := New(modelType)
		if err != nil {
			t.Errorf("Failed to create %s trainer: %v", modelType, err)
		}
		if trainer == nil {
			t.Errorf("Trainer is nil for type %s", modelType)
		}
	}
}

func TestUnigramTrainer_ViterbiSegmentation(t *testing.T) {
	// Test internal segmentation logic
	trainer := &unigramTrainer{
		config: &Config{VocabSize: 100},
		candidates: map[string]*candidatePiece{
			"h":     {piece: "h", freq: 10, score: -1.0},
			"e":     {piece: "e", freq: 10, score: -1.0},
			"l":     {piece: "l", freq: 20, score: -0.5},
			"o":     {piece: "o", freq: 10, score: -1.0},
			"hello": {piece: "hello", freq: 5, score: -2.0},
			"ll":    {piece: "ll", freq: 8, score: -1.5},
		},
	}

	segmentation := trainer.viterbiSegment("hello")
	if len(segmentation) == 0 {
		t.Error("Expected non-empty segmentation")
	}

	// Should prefer "hello" as single piece or reasonable subwords
	t.Logf("Segmentation of 'hello': %v", segmentation)
}

func TestUnigramTrainer_AddNGrams(t *testing.T) {
	trainer := &unigramTrainer{
		config:     &Config{VocabSize: 100},
		candidates: make(map[string]*candidatePiece),
	}

	sentences := []string{"hello", "hello", "world"}
	trainer.addNGrams(sentences, 2)

	// Should have bigrams like "he", "el", "ll", "lo", etc.
	if len(trainer.candidates) == 0 {
		t.Error("Expected bigrams to be added")
	}

	// Check for specific bigrams
	if _, exists := trainer.candidates["he"]; !exists {
		t.Error("Expected 'he' bigram to exist")
	}
}

func TestUnigramTrainer_PruneCandidates(t *testing.T) {
	trainer := &unigramTrainer{
		config: &Config{VocabSize: 100},
		candidates: map[string]*candidatePiece{
			"a": {piece: "a", freq: 10, score: -1.0},
			"b": {piece: "b", freq: 5, score: -2.0},
			"c": {piece: "c", freq: 15, score: -0.5},
			"d": {piece: "d", freq: 3, score: -3.0},
		},
	}

	trainer.pruneCandidates(2)

	if len(trainer.candidates) != 2 {
		t.Errorf("Expected 2 candidates after pruning, got %d", len(trainer.candidates))
	}

	// Should keep highest scoring pieces (c and a)
	if _, exists := trainer.candidates["c"]; !exists {
		t.Error("Expected 'c' to be kept (highest score)")
	}
	if _, exists := trainer.candidates["a"]; !exists {
		t.Error("Expected 'a' to be kept (second highest score)")
	}
}
