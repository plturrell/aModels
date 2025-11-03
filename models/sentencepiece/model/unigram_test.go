package model

import (
	"context"
	"testing"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
)

func TestNewUnigramModel(t *testing.T) {
	// Create a simple test model
	pieces := []*pb.ModelProto_SentencePiece{
		{Piece: strPtr("<unk>"), Score: floatPtr(0.0), Type: typePtr(pb.ModelProto_SentencePiece_UNKNOWN)},
		{Piece: strPtr("<s>"), Score: floatPtr(0.0), Type: typePtr(pb.ModelProto_SentencePiece_CONTROL)},
		{Piece: strPtr("</s>"), Score: floatPtr(0.0), Type: typePtr(pb.ModelProto_SentencePiece_CONTROL)},
		{Piece: strPtr("hello"), Score: floatPtr(-1.5), Type: typePtr(pb.ModelProto_SentencePiece_NORMAL)},
		{Piece: strPtr("world"), Score: floatPtr(-2.0), Type: typePtr(pb.ModelProto_SentencePiece_NORMAL)},
		{Piece: strPtr("h"), Score: floatPtr(-3.0), Type: typePtr(pb.ModelProto_SentencePiece_NORMAL)},
		{Piece: strPtr("e"), Score: floatPtr(-3.5), Type: typePtr(pb.ModelProto_SentencePiece_NORMAL)},
	}

	modelProto := &pb.ModelProto{
		Pieces: pieces,
		TrainerSpec: &pb.TrainerSpec{
			UnkId: int32Ptr(0),
			BosId: int32Ptr(1),
			EosId: int32Ptr(2),
		},
	}

	model, err := NewUnigramModel(modelProto)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	if model.GetPieceSize() != 7 {
		t.Errorf("Expected 7 pieces, got %d", model.GetPieceSize())
	}
}

func TestUnigramModel_GetPiece(t *testing.T) {
	model := createTestModel(t)

	piece, err := model.GetPiece(3)
	if err != nil {
		t.Fatalf("Failed to get piece: %v", err)
	}

	if piece != "hello" {
		t.Errorf("Expected 'hello', got '%s'", piece)
	}
}

func TestUnigramModel_Encode(t *testing.T) {
	model := createTestModel(t)

	ids, err := model.Encode(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Failed to encode: %v", err)
	}

	if len(ids) == 0 {
		t.Error("Expected non-empty result")
	}

	// Should encode as "hello" (id=3) since it's in vocabulary
	if len(ids) == 1 && ids[0] != 3 {
		t.Errorf("Expected id 3 for 'hello', got %d", ids[0])
	}
}

func TestUnigramModel_Decode(t *testing.T) {
	model := createTestModel(t)

	text, err := model.Decode(context.Background(), []int{3, 4})
	if err != nil {
		t.Fatalf("Failed to decode: %v", err)
	}

	expected := "helloworld"
	if text != expected {
		t.Errorf("Expected '%s', got '%s'", expected, text)
	}
}

func TestUnigramModel_EncodeAsPieces(t *testing.T) {
	model := createTestModel(t)

	pieces, err := model.EncodeAsPieces(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Failed to encode as pieces: %v", err)
	}

	if len(pieces) == 0 {
		t.Error("Expected non-empty result")
	}
}

func TestUnigramModel_EmptyText(t *testing.T) {
	model := createTestModel(t)

	ids, err := model.Encode(context.Background(), "")
	if err != nil {
		t.Fatalf("Failed to encode empty text: %v", err)
	}

	if len(ids) != 0 {
		t.Errorf("Expected empty result for empty text, got %d ids", len(ids))
	}
}

func TestUnigramModel_IsControl(t *testing.T) {
	model := createTestModel(t)

	if !model.IsControl(1) {
		t.Error("Expected id 1 (<s>) to be control token")
	}

	if model.IsControl(3) {
		t.Error("Expected id 3 (hello) to not be control token")
	}
}

func TestUnigramModel_IsUnknown(t *testing.T) {
	model := createTestModel(t)

	if !model.IsUnknown(0) {
		t.Error("Expected id 0 to be unknown token")
	}

	if model.IsUnknown(3) {
		t.Error("Expected id 3 to not be unknown token")
	}
}

// Helper functions

func createTestModel(t *testing.T) *UnigramModel {
	pieces := []*pb.ModelProto_SentencePiece{
		{Piece: strPtr("<unk>"), Score: floatPtr(0.0), Type: typePtr(pb.ModelProto_SentencePiece_UNKNOWN)},
		{Piece: strPtr("<s>"), Score: floatPtr(0.0), Type: typePtr(pb.ModelProto_SentencePiece_CONTROL)},
		{Piece: strPtr("</s>"), Score: floatPtr(0.0), Type: typePtr(pb.ModelProto_SentencePiece_CONTROL)},
		{Piece: strPtr("hello"), Score: floatPtr(-1.5), Type: typePtr(pb.ModelProto_SentencePiece_NORMAL)},
		{Piece: strPtr("world"), Score: floatPtr(-2.0), Type: typePtr(pb.ModelProto_SentencePiece_NORMAL)},
		{Piece: strPtr("h"), Score: floatPtr(-3.0), Type: typePtr(pb.ModelProto_SentencePiece_NORMAL)},
		{Piece: strPtr("e"), Score: floatPtr(-3.5), Type: typePtr(pb.ModelProto_SentencePiece_NORMAL)},
	}

	modelProto := &pb.ModelProto{
		Pieces: pieces,
		TrainerSpec: &pb.TrainerSpec{
			UnkId: int32Ptr(0),
			BosId: int32Ptr(1),
			EosId: int32Ptr(2),
		},
	}

	model, err := NewUnigramModel(modelProto)
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	return model
}

func strPtr(s string) *string {
	return &s
}

func floatPtr(f float32) *float32 {
	return &f
}

func typePtr(t pb.ModelProto_SentencePiece_Type) *pb.ModelProto_SentencePiece_Type {
	return &t
}

func int32Ptr(i int32) *int32 {
	return &i
}
