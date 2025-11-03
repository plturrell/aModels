package processor

import (
	"context"
	"math/rand"
	"testing"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
	"google.golang.org/protobuf/proto"
)

func createTestProcessor(t *testing.T, modelType pb.TrainerSpec_ModelType) *Processor {
	// Create a simple test model
	pieces := []*pb.ModelProto_SentencePiece{
		{Piece: proto.String("<unk>"), Score: proto.Float32(0.0), Type: pb.ModelProto_SentencePiece_UNKNOWN.Enum()},
		{Piece: proto.String("<s>"), Score: proto.Float32(0.0), Type: pb.ModelProto_SentencePiece_CONTROL.Enum()},
		{Piece: proto.String("</s>"), Score: proto.Float32(0.0), Type: pb.ModelProto_SentencePiece_CONTROL.Enum()},
		{Piece: proto.String("hello"), Score: proto.Float32(-1.0), Type: pb.ModelProto_SentencePiece_NORMAL.Enum()},
		{Piece: proto.String("world"), Score: proto.Float32(-1.5), Type: pb.ModelProto_SentencePiece_NORMAL.Enum()},
	}

	addDummyPrefix := true
	removeExtraWhitespaces := true
	escapeWhitespaces := true

	modelProto := &pb.ModelProto{
		Pieces: pieces,
		TrainerSpec: &pb.TrainerSpec{
			ModelType: &modelType,
		},
		NormalizerSpec: &pb.NormalizerSpec{
			AddDummyPrefix:         &addDummyPrefix,
			RemoveExtraWhitespaces: &removeExtraWhitespaces,
			EscapeWhitespaces:      &escapeWhitespaces,
		},
	}

	proc := New()
	
	// Load the full model to initialize normalizer properly
	if err := proc.loadFromProto(modelProto); err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	return proc
}

func TestSampleEncode_Unigram(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_UNIGRAM)
	ctx := context.Background()

	config := SamplingConfig{
		Alpha:      0.1,
		NumSamples: 3,
		Seed:       42,
	}

	results, err := proc.SampleEncode(ctx, "hello", config)
	if err != nil {
		t.Fatalf("SampleEncode failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one sample result")
	}

	for i, result := range results {
		t.Logf("Sample %d: %d tokens, %d pieces", i, len(result.IDs), len(result.Pieces))
	}
}

func TestSampleEncode_BPE(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_BPE)
	ctx := context.Background()

	config := SamplingConfig{
		Alpha:      0.1,
		NumSamples: 2,
		Seed:       42,
	}

	results, err := proc.SampleEncode(ctx, "hello", config)
	if err != nil {
		t.Fatalf("SampleEncode failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one sample result")
	}
}

func TestSampleEncode_WithoutReplacement(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_UNIGRAM)
	ctx := context.Background()

	config := SamplingConfig{
		Alpha:              0.5,
		NumSamples:         5,
		WithoutReplacement: true,
		Seed:               42,
	}

	results, err := proc.SampleEncode(ctx, "hello", config)
	if err != nil {
		t.Fatalf("SampleEncode failed: %v", err)
	}

	// Check for uniqueness
	seen := make(map[string]bool)
	for _, result := range results {
		key := piecesToKey(result.Pieces)
		if seen[key] {
			t.Error("Found duplicate sample with WOR enabled")
		}
		seen[key] = true
	}
}

func TestSampleEncode_IncludeBest(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_UNIGRAM)
	ctx := context.Background()

	config := SamplingConfig{
		Alpha:       0.1,
		NumSamples:  3,
		IncludeBest: true,
		Seed:        42,
	}

	results, err := proc.SampleEncode(ctx, "hello", config)
	if err != nil {
		t.Fatalf("SampleEncode failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one sample result")
	}

	// First result should be the best
	t.Logf("Best result: %d tokens", len(results[0].IDs))
}

func TestCalculateEntropy(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_UNIGRAM)
	ctx := context.Background()

	entropy, err := proc.CalculateEntropy(ctx, "hello", 0.1)
	if err != nil {
		t.Fatalf("CalculateEntropy failed: %v", err)
	}

	if entropy < 0 {
		t.Errorf("Expected non-negative entropy, got %f", entropy)
	}

	t.Logf("Entropy: %f", entropy)
}

func TestCalculateEntropy_NonUnigram(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_BPE)
	ctx := context.Background()

	entropy, err := proc.CalculateEntropy(ctx, "hello", 0.1)
	if err != nil {
		t.Fatalf("CalculateEntropy failed: %v", err)
	}

	// Should return 0 for non-Unigram models
	if entropy != 0 {
		t.Errorf("Expected 0 entropy for BPE model, got %f", entropy)
	}
}

func TestNBestEncode(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_UNIGRAM)
	ctx := context.Background()

	results, err := proc.NBestEncode(ctx, "hello", 5)
	if err != nil {
		t.Fatalf("NBestEncode failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one result")
	}

	t.Logf("Got %d N-best results", len(results))
}

func TestGumbelMax(t *testing.T) {
	// Test Gumbel-Max trick
	logProb := -1.0
	rng := rand.New(rand.NewSource(42))
	
	// Should not panic
	result := gumbelMax(logProb, rng)
	
	if result < logProb {
		t.Errorf("Gumbel-Max result should be >= logProb, got %f < %f", result, logProb)
	}
	
	t.Logf("Gumbel-Max result: %f", result)
}

func TestVocabSize(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_UNIGRAM)
	
	size := proc.VocabSize()
	if size != 5 {
		t.Errorf("Expected vocab size 5, got %d", size)
	}
}

func TestGetPieceAndScore(t *testing.T) {
	proc := createTestProcessor(t, pb.TrainerSpec_UNIGRAM)
	
	piece, score := proc.GetPieceAndScore(3)
	if piece != "hello" {
		t.Errorf("Expected piece 'hello', got '%s'", piece)
	}
	if score != -1.0 {
		t.Errorf("Expected score -1.0, got %f", score)
	}
}
