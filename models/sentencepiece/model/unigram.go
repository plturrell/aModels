package model

import (
	"context"
	"fmt"
	"math"
	"sort"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
)

// UnigramModel implements the Unigram language model for tokenization.
type UnigramModel struct {
	pieces []SentencePiece
	// Map from piece string to index for fast lookup
	pieceToID map[string]int
	// Special token IDs
	unkID int
	bosID int
	eosID int
	padID int
	// Min score for numerical stability
	minScore float32
}

// SentencePiece represents a vocabulary entry with its score.
type SentencePiece struct {
	Piece string
	Score float32
	Type  PieceType
}

// PieceType represents the type of a sentence piece.
type PieceType int

const (
	PieceTypeNormal PieceType = iota
	PieceTypeUnknown
	PieceTypeControl
	PieceTypeUserDefined
	PieceTypeByte
	PieceTypeUnused
)

// NewUnigramModel creates a new Unigram model from a protobuf ModelProto.
func NewUnigramModel(modelProto *pb.ModelProto) (*UnigramModel, error) {
	if modelProto == nil {
		return nil, fmt.Errorf("model proto is nil")
	}

	if len(modelProto.Pieces) == 0 {
		return nil, fmt.Errorf("model has no vocabulary pieces")
	}

	model := &UnigramModel{
		pieces:    make([]SentencePiece, 0, len(modelProto.Pieces)),
		pieceToID: make(map[string]int),
		unkID:     0,
		bosID:     1,
		eosID:     2,
		padID:     -1,
		minScore:  math.MaxFloat32,
	}

	// Load special token IDs from trainer spec
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

		// Track minimum score for numerical stability
		if sp.Score < model.minScore && sp.Type == PieceTypeNormal {
			model.minScore = sp.Score
		}
	}

	return model, nil
}

// convertPieceType converts protobuf piece type to internal type.
func convertPieceType(pbType pb.ModelProto_SentencePiece_Type) PieceType {
	switch pbType {
	case pb.ModelProto_SentencePiece_NORMAL:
		return PieceTypeNormal
	case pb.ModelProto_SentencePiece_UNKNOWN:
		return PieceTypeUnknown
	case pb.ModelProto_SentencePiece_CONTROL:
		return PieceTypeControl
	case pb.ModelProto_SentencePiece_USER_DEFINED:
		return PieceTypeUserDefined
	case pb.ModelProto_SentencePiece_BYTE:
		return PieceTypeByte
	case pb.ModelProto_SentencePiece_UNUSED:
		return PieceTypeUnused
	default:
		return PieceTypeNormal
	}
}

// Encode converts text into token IDs using Viterbi algorithm.
func (m *UnigramModel) Encode(ctx context.Context, text string) ([]int, error) {
	if len(text) == 0 {
		return []int{}, nil
	}

	// Build lattice for Viterbi decoding
	lattice := m.buildLattice(text)
	
	// Find best path through lattice
	path := m.viterbi(lattice)
	
	// Convert path to token IDs
	ids := make([]int, 0, len(path))
	for _, node := range path {
		if node.id >= 0 {
			ids = append(ids, node.id)
		}
	}

	return ids, nil
}

// Decode converts token IDs back into text.
func (m *UnigramModel) Decode(ctx context.Context, ids []int) (string, error) {
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
func (m *UnigramModel) EncodeAsPieces(ctx context.Context, text string) ([]string, error) {
	if len(text) == 0 {
		return []string{}, nil
	}

	lattice := m.buildLattice(text)
	path := m.viterbi(lattice)
	
	pieces := make([]string, 0, len(path))
	for _, node := range path {
		if node.id >= 0 {
			pieces = append(pieces, m.pieces[node.id].Piece)
		}
	}

	return pieces, nil
}

// GetPieceSize returns the vocabulary size.
func (m *UnigramModel) GetPieceSize() int {
	return len(m.pieces)
}

// GetPiece returns the piece string for a given ID.
func (m *UnigramModel) GetPiece(id int) (string, error) {
	if id < 0 || id >= len(m.pieces) {
		return "", fmt.Errorf("invalid piece ID: %d", id)
	}
	return m.pieces[id].Piece, nil
}

// GetScore returns the score for a given piece ID.
func (m *UnigramModel) GetScore(id int) (float32, error) {
	if id < 0 || id >= len(m.pieces) {
		return 0, fmt.Errorf("invalid piece ID: %d", id)
	}
	return m.pieces[id].Score, nil
}

// IsControl returns true if the piece is a control token.
func (m *UnigramModel) IsControl(id int) bool {
	if id < 0 || id >= len(m.pieces) {
		return false
	}
	return m.pieces[id].Type == PieceTypeControl
}

// IsUnknown returns true if the piece is the unknown token.
func (m *UnigramModel) IsUnknown(id int) bool {
	return id == m.unkID
}

// IsByte returns true if the piece represents a byte fallback.
func (m *UnigramModel) IsByte(id int) bool {
	if id < 0 || id >= len(m.pieces) {
		return false
	}
	return m.pieces[id].Type == PieceTypeByte
}

// latticeNode represents a node in the tokenization lattice.
type latticeNode struct {
	pos   int     // Position in text
	id    int     // Piece ID (-1 for BOS/EOS)
	score float32 // Cumulative score
	prev  *latticeNode
}

// buildLattice constructs a lattice of possible tokenizations.
func (m *UnigramModel) buildLattice(text string) [][]*latticeNode {
	textRunes := []rune(text)
	n := len(textRunes)
	
	// lattice[i] contains all nodes ending at position i
	lattice := make([][]*latticeNode, n+1)
	
	// Initialize with BOS node
	lattice[0] = []*latticeNode{{pos: 0, id: -1, score: 0, prev: nil}}
	
	// Build lattice
	for i := 0; i < n; i++ {
		if len(lattice[i]) == 0 {
			continue
		}
		
		// Try all possible pieces starting at position i
		for length := 1; length <= n-i; length++ {
			piece := string(textRunes[i : i+length])
			
			if pieceID, ok := m.pieceToID[piece]; ok {
				// Found a matching piece
				pieceScore := m.pieces[pieceID].Score
				
				// Add node for each predecessor
				for _, prevNode := range lattice[i] {
					newNode := &latticeNode{
						pos:   i + length,
						id:    pieceID,
						score: prevNode.score + pieceScore,
						prev:  prevNode,
					}
					lattice[i+length] = append(lattice[i+length], newNode)
				}
			}
		}
		
		// Handle unknown character with UNK token
		if len(lattice[i+1]) == 0 {
			for _, prevNode := range lattice[i] {
				newNode := &latticeNode{
					pos:   i + 1,
					id:    m.unkID,
					score: prevNode.score + m.minScore,
					prev:  prevNode,
				}
				lattice[i+1] = append(lattice[i+1], newNode)
			}
		}
	}
	
	return lattice
}

// viterbi finds the best path through the lattice.
func (m *UnigramModel) viterbi(lattice [][]*latticeNode) []*latticeNode {
	if len(lattice) == 0 {
		return nil
	}
	
	// Find the best final node
	finalNodes := lattice[len(lattice)-1]
	if len(finalNodes) == 0 {
		return nil
	}
	
	// Sort by score (higher is better)
	sort.Slice(finalNodes, func(i, j int) bool {
		return finalNodes[i].score > finalNodes[j].score
	})
	
	bestNode := finalNodes[0]
	
	// Backtrack to build path
	path := make([]*latticeNode, 0)
	for node := bestNode; node != nil && node.prev != nil; node = node.prev {
		path = append(path, node)
	}
	
	// Reverse path to get forward order
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	
	return path
}
