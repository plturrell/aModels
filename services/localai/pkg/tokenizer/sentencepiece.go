package tokenizer

import (
	"context"
	"fmt"
	"sync"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/processor"
)

// SentencePieceTokenizer provides high-performance subword tokenization for LocalAI.
type SentencePieceTokenizer struct {
	processor *processor.Processor
	mu        sync.RWMutex
	modelPath string
}

// NewSentencePieceTokenizer creates a new tokenizer from a model file.
func NewSentencePieceTokenizer(modelPath string) (*SentencePieceTokenizer, error) {
	proc := processor.New()
	if err := proc.Load(modelPath); err != nil {
		return nil, fmt.Errorf("failed to load sentencepiece model: %w", err)
	}

	return &SentencePieceTokenizer{
		processor: proc,
		modelPath: modelPath,
	}, nil
}

// Encode tokenizes text into token IDs.
func (t *SentencePieceTokenizer) Encode(text string) ([]int, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.processor == nil {
		return nil, fmt.Errorf("tokenizer not initialized")
	}

	return t.processor.Encode(context.Background(), text)
}

// EncodeAsTokens tokenizes text into token strings.
func (t *SentencePieceTokenizer) EncodeAsTokens(text string) ([]string, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.processor == nil {
		return nil, fmt.Errorf("tokenizer not initialized")
	}

	return t.processor.EncodeAsPieces(context.Background(), text)
}

// Decode converts token IDs back to text.
func (t *SentencePieceTokenizer) Decode(ids []int) (string, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.processor == nil {
		return "", fmt.Errorf("tokenizer not initialized")
	}

	return t.processor.Decode(context.Background(), ids)
}

// VocabSize returns the vocabulary size.
func (t *SentencePieceTokenizer) VocabSize() int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.processor == nil {
		return 0
	}

	return t.processor.VocabSize()
}

// GetToken returns the token string for a given ID.
func (t *SentencePieceTokenizer) GetToken(id int) (string, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.processor == nil {
		return "", fmt.Errorf("tokenizer not initialized")
	}

	token, _ := t.processor.GetPieceAndScore(id)
	if token == "" {
		return "", fmt.Errorf("invalid token ID: %d", id)
	}

	return token, nil
}

// SampleEncode performs sampling-based encoding for diverse tokenizations.
// Useful for data augmentation and robustness testing.
func (t *SentencePieceTokenizer) SampleEncode(text string, alpha float64, numSamples int) ([][]int, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.processor == nil {
		return nil, fmt.Errorf("tokenizer not initialized")
	}

	config := processor.SamplingConfig{
		Alpha:      alpha,
		NumSamples: numSamples,
	}

	results, err := t.processor.SampleEncode(context.Background(), text, config)
	if err != nil {
		return nil, err
	}

	samples := make([][]int, len(results))
	for i, result := range results {
		samples[i] = result.IDs
	}

	return samples, nil
}

// Close releases resources (no-op for Go implementation).
func (t *SentencePieceTokenizer) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.processor = nil
	return nil
}
