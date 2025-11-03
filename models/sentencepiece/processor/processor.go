package processor

import (
	"context"
	"errors"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/normalizer"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

// Common errors
var (
	ErrModelNotLoaded = errors.New("model not loaded")
)

// Processor is the main interface for SentencePiece operations.
type Processor struct {
	model      model.Model
	normalizer *normalizer.Normalizer
	modelType  string // UNIGRAM, BPE, WORD, CHAR
}

// New creates a new Processor instance.
func New() *Processor {
	return &Processor{}
}

// Load loads a SentencePiece model from a file.
func (p *Processor) Load(modelPath string) error {
	return p.LoadModel(modelPath)
}

// Encode converts text into token IDs.
func (p *Processor) Encode(ctx context.Context, text string) ([]int, error) {
	if p.model == nil {
		return nil, errors.New("model not loaded")
	}
	return p.model.Encode(ctx, text)
}

// Decode converts token IDs back into text.
func (p *Processor) Decode(ctx context.Context, ids []int) (string, error) {
	if p.model == nil {
		return "", errors.New("model not loaded")
	}
	return p.model.Decode(ctx, ids)
}

// EncodeAsPieces converts text into subword pieces.
func (p *Processor) EncodeAsPieces(ctx context.Context, text string) ([]string, error) {
	if p.model == nil {
		return nil, errors.New("model not loaded")
	}
	return p.model.EncodeAsPieces(ctx, text)
}

// GetPieceSize returns the vocabulary size.
func (p *Processor) GetPieceSize() int {
	if p.model == nil {
		return 0
	}
	return p.model.GetPieceSize()
}

// VocabSize returns the vocabulary size (alias for GetPieceSize).
func (p *Processor) VocabSize() int {
	return p.GetPieceSize()
}

// GetPieceAndScore returns the piece string and score for a given vocabulary ID.
func (p *Processor) GetPieceAndScore(id int) (string, float32) {
	if p.model == nil {
		return "", 0.0
	}
	piece, err := p.model.GetPiece(id)
	if err != nil {
		return "", 0.0
	}
	score, err := p.model.GetScore(id)
	if err != nil {
		return piece, 0.0
	}
	return piece, score
}
