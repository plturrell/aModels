package glove

import (
	"context"
	"fmt"
	"sync"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/processor"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/trainer"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

// SentencePieceTokenizer provides subword tokenization
type SentencePieceTokenizer struct {
	processor *processor.Processor
	mu        sync.RWMutex
	vocab     map[string]int
	idToWord  map[int]string
}

// NewSentencePieceTokenizer creates a new SentencePiece tokenizer from a trained model file.
func NewSentencePieceTokenizer(modelPath string) (*SentencePieceTokenizer, error) {
	proc := processor.New()
	if err := proc.Load(modelPath); err != nil {
		return nil, fmt.Errorf("failed to load sentencepiece model from %s: %w", modelPath, err)
	}

	sp := &SentencePieceTokenizer{
		processor: proc,
		vocab:     make(map[string]int),
		idToWord:  make(map[int]string),
	}

	// Build vocabulary from the loaded SentencePiece model
	sp.buildVocabulary()

	return sp, nil
}

// TrainSentencePiece trains a new SentencePiece model from a corpus using the Go trainer.
func TrainSentencePiece(ctx context.Context, corpus []string, modelPrefix string, vocabSize int) error {
	// Create trainer configuration
	config := &trainer.Config{
		InputData:         corpus,
		ModelType:         model.TypeBPE, // Default to BPE
		VocabSize:         vocabSize,
		CharacterCoverage: 0.9995,
	}

	// Create trainer
	tr, err := trainer.New(model.TypeBPE)
	if err != nil {
		return fmt.Errorf("failed to create trainer: %w", err)
	}

	// Train the model
	trainedModel, err := tr.Train(ctx, config)
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	// Save the model (implementation would save to modelPrefix.model)
	_ = trainedModel // TODO: Implement model saving to file
	// For now, models are saved via HANA DB in the trainer

	return nil
}

// Tokenize performs subword tokenization.
func (sp *SentencePieceTokenizer) Tokenize(text string) ([]string, error) {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	if sp.processor == nil {
		return tokenize(text), nil // Fallback to default tokenizer
	}

	ids, err := sp.processor.Encode(context.Background(), text)
	if err != nil {
		return tokenize(text), fmt.Errorf("sp encoding failed: %w", err)
	}

	tokens := make([]string, len(ids))
	for i, id := range ids {
		piece, ok := sp.idToWord[id]
		if !ok {
			p, _ := sp.processor.GetPieceAndScore(id)
			if p != "" {
				sp.idToWord[id] = p
				sp.vocab[p] = id
				piece = p
			}
		}
		tokens[i] = piece
	}

	return tokens, nil
}

// TokenizeWithIDs returns tokens and their SentencePiece IDs.
func (sp *SentencePieceTokenizer) TokenizeWithIDs(text string) ([]string, []int, error) {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	if sp.processor == nil {
		tokens := tokenize(text)
		ids := make([]int, len(tokens))
		for i := range ids {
			ids[i] = -1 // Unknown ID for fallback tokens
		}
		return tokens, ids, nil
	}

	ids, err := sp.processor.Encode(context.Background(), text)
	if err != nil {
		return nil, nil, fmt.Errorf("sp encoding to IDs failed: %w", err)
	}

	tokens := make([]string, len(ids))
	for i, id := range ids {
		piece, ok := sp.idToWord[id]
		if !ok {
			p, _ := sp.processor.GetPieceAndScore(id)
			if p != "" {
				sp.idToWord[id] = p
				sp.vocab[p] = id
				piece = p
			}
		}
		tokens[i] = piece
	}

	return tokens, append([]int(nil), ids...), nil
}

// buildVocabulary populates the internal vocab map from the SentencePiece model.
func (sp *SentencePieceTokenizer) buildVocabulary() {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	vocabSize := sp.processor.VocabSize()
	for i := 0; i < vocabSize; i++ {
		piece, _ := sp.processor.GetPieceAndScore(i)
		if piece == "" {
			continue
		}
		sp.vocab[piece] = i
		sp.idToWord[i] = piece
	}
}

// VocabSize returns the vocabulary size.
func (sp *SentencePieceTokenizer) VocabSize() int {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	return len(sp.vocab)
}

// Close releases SentencePiece model resources.
func (sp *SentencePieceTokenizer) Close() {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	// Go processor doesn't require explicit cleanup
	sp.processor = nil
}
