package model

import (
	"context"
	"fmt"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
)

// BPEModel implements Byte Pair Encoding tokenization.
type BPEModel struct {
	pieces    []SentencePiece
	pieceToID map[string]int
	// Merge rules: pairs of pieces that can be merged
	merges    []mergePair
	mergeRank map[mergePair]int
	// Special token IDs
	unkID int
	bosID int
	eosID int
	padID int
}

// mergePair represents a pair of pieces that can be merged.
type mergePair struct {
	left  string
	right string
}

// NewBPEModel creates a new BPE model from a protobuf ModelProto.
func NewBPEModel(modelProto *pb.ModelProto) (*BPEModel, error) {
	if modelProto == nil {
		return nil, fmt.Errorf("model proto is nil")
	}

	if len(modelProto.Pieces) == 0 {
		return nil, fmt.Errorf("model has no vocabulary pieces")
	}

	model := &BPEModel{
		pieces:    make([]SentencePiece, 0, len(modelProto.Pieces)),
		pieceToID: make(map[string]int),
		merges:    make([]mergePair, 0),
		mergeRank: make(map[mergePair]int),
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

	// Build merge rules from vocabulary order
	// In BPE, pieces are ordered by merge priority
	model.buildMergeRules()

	return model, nil
}

// buildMergeRules constructs merge rules from vocabulary.
func (m *BPEModel) buildMergeRules() {
	// BPE merges are implicit in the vocabulary order
	// Later pieces in vocab represent merged pairs
	for i, piece := range m.pieces {
		if piece.Type != PieceTypeNormal {
			continue
		}

		// Try to split piece into two parts
		for splitPos := 1; splitPos < len(piece.Piece); splitPos++ {
			left := piece.Piece[:splitPos]
			right := piece.Piece[splitPos:]

			// Check if both parts exist in vocabulary
			if _, leftExists := m.pieceToID[left]; leftExists {
				if _, rightExists := m.pieceToID[right]; rightExists {
					pair := mergePair{left: left, right: right}
					m.merges = append(m.merges, pair)
					m.mergeRank[pair] = i
					break
				}
			}
		}
	}
}

// Encode converts text into token IDs using BPE algorithm.
func (m *BPEModel) Encode(ctx context.Context, text string) ([]int, error) {
	if len(text) == 0 {
		return []int{}, nil
	}

	// Start with character-level tokenization
	tokens := m.initializeTokens(text)

	// Apply BPE merges
	tokens = m.applyMerges(tokens)

	// Convert to IDs
	ids := make([]int, 0, len(tokens))
	for _, token := range tokens {
		if id, exists := m.pieceToID[token]; exists {
			ids = append(ids, id)
		} else {
			ids = append(ids, m.unkID)
		}
	}

	return ids, nil
}

// initializeTokens splits text into initial character tokens.
func (m *BPEModel) initializeTokens(text string) []string {
	tokens := make([]string, 0)
	for _, r := range text {
		tokens = append(tokens, string(r))
	}
	return tokens
}

// applyMerges applies BPE merge rules to tokens.
func (m *BPEModel) applyMerges(tokens []string) []string {
	if len(tokens) <= 1 {
		return tokens
	}

	// Keep merging until no more merges possible
	for {
		// Find the best merge (lowest rank = highest priority)
		bestMerge := -1
		bestRank := len(m.pieces) + 1

		for i := 0; i < len(tokens)-1; i++ {
			pair := mergePair{left: tokens[i], right: tokens[i+1]}
			if rank, exists := m.mergeRank[pair]; exists {
				if rank < bestRank {
					bestRank = rank
					bestMerge = i
				}
			}
		}

		// No more merges possible
		if bestMerge == -1 {
			break
		}

		// Apply the merge
		merged := tokens[bestMerge] + tokens[bestMerge+1]
		newTokens := make([]string, 0, len(tokens)-1)
		newTokens = append(newTokens, tokens[:bestMerge]...)
		newTokens = append(newTokens, merged)
		newTokens = append(newTokens, tokens[bestMerge+2:]...)
		tokens = newTokens
	}

	return tokens
}

// Decode converts token IDs back into text.
func (m *BPEModel) Decode(ctx context.Context, ids []int) (string, error) {
	result := ""
	for _, id := range ids {
		if id < 0 || id >= len(m.pieces) {
			return "", fmt.Errorf("invalid token ID: %d", id)
		}
		result += m.pieces[id].Piece
	}
	return result, nil
}

// EncodeAsPieces converts text into subword pieces.
func (m *BPEModel) EncodeAsPieces(ctx context.Context, text string) ([]string, error) {
	if len(text) == 0 {
		return []string{}, nil
	}

	tokens := m.initializeTokens(text)
	tokens = m.applyMerges(tokens)

	return tokens, nil
}

// GetPieceSize returns the vocabulary size.
func (m *BPEModel) GetPieceSize() int {
	return len(m.pieces)
}

// GetPiece returns the piece string for a given ID.
func (m *BPEModel) GetPiece(id int) (string, error) {
	if id < 0 || id >= len(m.pieces) {
		return "", fmt.Errorf("invalid piece ID: %d", id)
	}
	return m.pieces[id].Piece, nil
}

// GetScore returns the score for a given piece ID.
func (m *BPEModel) GetScore(id int) (float32, error) {
	if id < 0 || id >= len(m.pieces) {
		return 0, fmt.Errorf("invalid piece ID: %d", id)
	}
	return m.pieces[id].Score, nil
}

// IsControl returns true if the piece is a control token.
func (m *BPEModel) IsControl(id int) bool {
	if id < 0 || id >= len(m.pieces) {
		return false
	}
	return m.pieces[id].Type == PieceTypeControl
}

// IsUnknown returns true if the piece is the unknown token.
func (m *BPEModel) IsUnknown(id int) bool {
	return id == m.unkID
}

// IsByte returns true if the piece represents a byte fallback.
func (m *BPEModel) IsByte(id int) bool {
	if id < 0 || id >= len(m.pieces) {
		return false
	}
	return m.pieces[id].Type == PieceTypeByte
}
