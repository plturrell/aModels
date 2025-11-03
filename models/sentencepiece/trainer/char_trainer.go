package trainer

import (
	"context"
	"fmt"
	"sort"
	"unicode/utf8"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

// charTrainer implements the character-based trainer.
type charTrainer struct {
	config   *Config
	charFreq map[string]int
	vocab    []charEntry
}

type charEntry struct {
	char  string
	freq  int
	score float32
}

// Train trains a Char model from input data.
func (t *charTrainer) Train(ctx context.Context, config *Config) (model.Model, error) {
	t.config = config
	t.charFreq = make(map[string]int)

	// Load training data
	sentences, err := t.loadTrainingData()
	if err != nil {
		return nil, fmt.Errorf("failed to load training data: %w", err)
	}

	if len(sentences) == 0 {
		return nil, fmt.Errorf("no training data provided")
	}

	// Count character frequencies
	t.countCharacters(sentences)

	// Build vocabulary
	t.buildVocabulary()

	// Build model
	return t.buildModel()
}

// loadTrainingData loads and preprocesses training sentences.
func (t *charTrainer) loadTrainingData() ([]string, error) {
	var sentences []string

	if len(t.config.InputData) > 0 {
		sentences = append(sentences, t.config.InputData...)
	}

	// Apply max sentence length limit
	if t.config.MaxSentenceLength > 0 {
		filtered := make([]string, 0, len(sentences))
		for _, s := range sentences {
			if utf8.RuneCountInString(s) <= t.config.MaxSentenceLength {
				filtered = append(filtered, s)
			}
		}
		sentences = filtered
	}

	return sentences, nil
}

// countCharacters counts character frequencies in training data.
func (t *charTrainer) countCharacters(sentences []string) {
	for _, sentence := range sentences {
		for _, r := range sentence {
			char := string(r)
			t.charFreq[char]++
		}
	}
}

// buildVocabulary builds the vocabulary from character frequencies.
func (t *charTrainer) buildVocabulary() {
	// Convert to slice
	t.vocab = make([]charEntry, 0, len(t.charFreq))
	for char, freq := range t.charFreq {
		t.vocab = append(t.vocab, charEntry{
			char: char,
			freq: freq,
		})
	}

	// Sort by frequency (descending)
	sort.Slice(t.vocab, func(i, j int) bool {
		return t.vocab[i].freq > t.vocab[j].freq
	})

	// Keep top N characters (usually all characters are kept)
	if len(t.vocab) > t.config.VocabSize {
		t.vocab = t.vocab[:t.config.VocabSize]
	}

	// Assign scores
	for i := range t.vocab {
		t.vocab[i].score = float32(-i)
	}
}

// buildModel constructs the final Char model.
func (t *charTrainer) buildModel() (model.Model, error) {
	pieces := make([]*pb.ModelProto_SentencePiece, 0, len(t.vocab)+3)

	// Add special tokens
	unkType := pb.ModelProto_SentencePiece_UNKNOWN
	controlType := pb.ModelProto_SentencePiece_CONTROL
	normalType := pb.ModelProto_SentencePiece_NORMAL

	pieces = append(pieces, &pb.ModelProto_SentencePiece{
		Piece: strPtr("<unk>"),
		Score: floatPtr(0.0),
		Type:  &unkType,
	})
	pieces = append(pieces, &pb.ModelProto_SentencePiece{
		Piece: strPtr("<s>"),
		Score: floatPtr(0.0),
		Type:  &controlType,
	})
	pieces = append(pieces, &pb.ModelProto_SentencePiece{
		Piece: strPtr("</s>"),
		Score: floatPtr(0.0),
		Type:  &controlType,
	})

	// Add vocabulary
	for _, entry := range t.vocab {
		pieces = append(pieces, &pb.ModelProto_SentencePiece{
			Piece: strPtr(entry.char),
			Score: floatPtr(entry.score),
			Type:  &normalType,
		})
	}

	// Build trainer spec
	modelType := pb.TrainerSpec_CHAR
	trainerSpec := &pb.TrainerSpec{
		ModelType:         &modelType,
		VocabSize:         int32Ptr(int32(t.config.VocabSize)),
		CharacterCoverage: floatPtr(float32(t.config.CharacterCoverage)),
		UnkId:             int32Ptr(0),
		BosId:             int32Ptr(1),
		EosId:             int32Ptr(2),
		PadId:             int32Ptr(-1),
	}

	// Build normalizer spec
	normalizerSpec := &pb.NormalizerSpec{
		AddDummyPrefix:         boolPtr(false), // Char model doesn't use dummy prefix
		RemoveExtraWhitespaces: boolPtr(false), // Keep all characters
		EscapeWhitespaces:      boolPtr(false),
	}

	// Create ModelProto
	modelProto := &pb.ModelProto{
		Pieces:         pieces,
		TrainerSpec:    trainerSpec,
		NormalizerSpec: normalizerSpec,
	}

	// Build and return Char model
	return model.NewCharModel(modelProto)
}
