package model

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
)

// WordModel implements word-based tokenization.
type WordModel struct {
	pieces    []SentencePiece
	pieceToID map[string]int
	unkID     int
	bosID     int
	eosID     int
	padID     int
}

// NewWordModel creates a new Word model from a protobuf ModelProto.
func NewWordModel(modelProto *pb.ModelProto) (*WordModel, error) {
	if modelProto == nil {
		return nil, fmt.Errorf("model proto is nil")
	}

	if len(modelProto.Pieces) == 0 {
		return nil, fmt.Errorf("model has no vocabulary pieces")
	}

	model := &WordModel{
		pieces:    make([]SentencePiece, 0, len(modelProto.Pieces)),
		pieceToID: make(map[string]int),
		unkID:     0,
		bosID:     1,
		eosID:     2,
		padID:     -1,
	}

	// Load special token IDs
	if modelProto.TrainerSpec != nil {
		model.unkID = int(modelProto.TrainerSpec.GetUnkId())
		model.bosID = int(modelProto.TrainerSpec.GetBosId())
		model.eosID = int(modelProto.TrainerSpec.GetEosId())
		model.padID = int(modelProto.TrainerSpec.GetPadId())
	}

	// Load vocabulary pieces
	for i, piece := range modelProto.Pieces {
		if piece.Piece == nil {
			continue
		}

		sp := SentencePiece{
			Piece: piece.GetPiece(),
			Score: piece.GetScore(),
			Type:  convertPieceType(piece.GetType()),
		}

		model.pieces = append(model.pieces, sp)
		model.pieceToID[sp.Piece] = i
	}

	return model, nil
}

// Encode converts text into token IDs using word boundaries.
func (m *WordModel) Encode(ctx context.Context, text string) ([]int, error) {
	if len(text) == 0 {
		return []int{}, nil
	}

	// Split into words
	words := m.splitWords(text)

	// Convert to IDs
	ids := make([]int, 0, len(words))
	for _, word := range words {
		if id, exists := m.pieceToID[word]; exists {
			ids = append(ids, id)
		} else {
			// Try lowercase
			lowerWord := strings.ToLower(word)
			if id, exists := m.pieceToID[lowerWord]; exists {
				ids = append(ids, id)
			} else {
				ids = append(ids, m.unkID)
			}
		}
	}

	return ids, nil
}

// splitWords splits text into words based on whitespace and punctuation.
func (m *WordModel) splitWords(text string) []string {
	var words []string
	var currentWord strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if currentWord.Len() > 0 {
				words = append(words, currentWord.String())
				currentWord.Reset()
			}
			// Include punctuation as separate tokens
			if unicode.IsPunct(r) {
				words = append(words, string(r))
			}
		} else {
			currentWord.WriteRune(r)
		}
	}

	if currentWord.Len() > 0 {
		words = append(words, currentWord.String())
	}

	return words
}

// Decode converts token IDs back into text.
func (m *WordModel) Decode(ctx context.Context, ids []int) (string, error) {
	var result strings.Builder
	for i, id := range ids {
		if id < 0 || id >= len(m.pieces) {
			return "", fmt.Errorf("invalid token ID: %d", id)
		}
		if i > 0 {
			result.WriteString(" ")
		}
		result.WriteString(m.pieces[id].Piece)
	}
	return result.String(), nil
}

// EncodeAsPieces converts text into word pieces.
func (m *WordModel) EncodeAsPieces(ctx context.Context, text string) ([]string, error) {
	if len(text) == 0 {
		return []string{}, nil
	}

	return m.splitWords(text), nil
}

// GetPieceSize returns the vocabulary size.
func (m *WordModel) GetPieceSize() int {
	return len(m.pieces)
}

// GetPiece returns the piece string for a given ID.
func (m *WordModel) GetPiece(id int) (string, error) {
	if id < 0 || id >= len(m.pieces) {
		return "", fmt.Errorf("invalid piece ID: %d", id)
	}
	return m.pieces[id].Piece, nil
}

// GetScore returns the score for a given piece ID.
func (m *WordModel) GetScore(id int) (float32, error) {
	if id < 0 || id >= len(m.pieces) {
		return 0, fmt.Errorf("invalid piece ID: %d", id)
	}
	return m.pieces[id].Score, nil
}

// IsControl returns true if the piece is a control token.
func (m *WordModel) IsControl(id int) bool {
	if id < 0 || id >= len(m.pieces) {
		return false
	}
	return m.pieces[id].Type == PieceTypeControl
}

// IsUnknown returns true if the piece is the unknown token.
func (m *WordModel) IsUnknown(id int) bool {
	return id == m.unkID
}

// IsByte returns true if the piece represents a byte fallback.
func (m *WordModel) IsByte(id int) bool {
	if id < 0 || id >= len(m.pieces) {
		return false
	}
	return m.pieces[id].Type == PieceTypeByte
}
