package model

import (
	"context"
)

// Model defines the interface for SentencePiece tokenization models.
type Model interface {
	// Encode converts text into token IDs.
	Encode(ctx context.Context, text string) ([]int, error)

	// Decode converts token IDs back into text.
	Decode(ctx context.Context, ids []int) (string, error)

	// EncodeAsPieces converts text into subword pieces.
	EncodeAsPieces(ctx context.Context, text string) ([]string, error)

	// GetPieceSize returns the vocabulary size.
	GetPieceSize() int

	// GetPiece returns the piece string for a given ID.
	GetPiece(id int) (string, error)

	// GetScore returns the score for a given piece ID.
	GetScore(id int) (float32, error)

	// IsControl returns true if the piece is a control token.
	IsControl(id int) bool

	// IsUnknown returns true if the piece is the unknown token.
	IsUnknown(id int) bool

	// IsByte returns true if the piece represents a byte fallback.
	IsByte(id int) bool
}

// Type represents the model algorithm type.
type Type int

const (
	TypeUnigram Type = iota
	TypeBPE
	TypeWord
	TypeChar
)

func (t Type) String() string {
	switch t {
	case TypeUnigram:
		return "UNIGRAM"
	case TypeBPE:
		return "BPE"
	case TypeWord:
		return "WORD"
	case TypeChar:
		return "CHAR"
	default:
		return "UNKNOWN"
	}
}
