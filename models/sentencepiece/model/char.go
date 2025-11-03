package model

import (
	"context"
	"fmt"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
)

// CharModel implements character-based tokenization.
type CharModel struct {
	pieces    []SentencePiece
	pieceToID map[string]int
	unkID     int
	bosID     int
	eosID     int
	padID     int
}

// NewCharModel creates a new Char model from a protobuf ModelProto.
func NewCharModel(modelProto *pb.ModelProto) (*CharModel, error) {
	if modelProto == nil {
		return nil, fmt.Errorf("model proto is nil")
	}

	if len(modelProto.Pieces) == 0 {
		return nil, fmt.Errorf("model has no vocabulary pieces")
	}

	model := &CharModel{
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

// Encode converts text into token IDs (one per character).
func (m *CharModel) Encode(ctx context.Context, text string) ([]int, error) {
	if len(text) == 0 {
		return []int{}, nil
	}

	ids := make([]int, 0, len(text))
	for _, r := range text {
		char := string(r)
		if id, exists := m.pieceToID[char]; exists {
			ids = append(ids, id)
		} else {
			ids = append(ids, m.unkID)
		}
	}

	return ids, nil
}

// Decode converts token IDs back into text.
func (m *CharModel) Decode(ctx context.Context, ids []int) (string, error) {
	result := ""
	for _, id := range ids {
		if id < 0 || id >= len(m.pieces) {
			return "", fmt.Errorf("invalid token ID: %d", id)
		}
		result += m.pieces[id].Piece
	}
	return result, nil
}

// EncodeAsPieces converts text into character pieces.
func (m *CharModel) EncodeAsPieces(ctx context.Context, text string) ([]string, error) {
	if len(text) == 0 {
		return []string{}, nil
	}

	pieces := make([]string, 0, len(text))
	for _, r := range text {
		pieces = append(pieces, string(r))
	}

	return pieces, nil
}

// GetPieceSize returns the vocabulary size.
func (m *CharModel) GetPieceSize() int {
	return len(m.pieces)
}

// GetPiece returns the piece string for a given ID.
func (m *CharModel) GetPiece(id int) (string, error) {
	if id < 0 || id >= len(m.pieces) {
		return "", fmt.Errorf("invalid piece ID: %d", id)
	}
	return m.pieces[id].Piece, nil
}

// GetScore returns the score for a given piece ID.
func (m *CharModel) GetScore(id int) (float32, error) {
	if id < 0 || id >= len(m.pieces) {
		return 0, fmt.Errorf("invalid piece ID: %d", id)
	}
	return m.pieces[id].Score, nil
}

// IsControl returns true if the piece is a control token.
func (m *CharModel) IsControl(id int) bool {
	if id < 0 || id >= len(m.pieces) {
		return false
	}
	return m.pieces[id].Type == PieceTypeControl
}

// IsUnknown returns true if the piece is the unknown token.
func (m *CharModel) IsUnknown(id int) bool {
	return id == m.unkID
}

// IsByte returns true if the piece represents a byte fallback.
func (m *CharModel) IsByte(id int) bool {
	if id < 0 || id >= len(m.pieces) {
		return false
	}
	return m.pieces[id].Type == PieceTypeByte
}
