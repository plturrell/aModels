package trainer

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

// wordTrainer implements the word-based trainer.
type wordTrainer struct {
	config    *Config
	wordFreq  map[string]int
	vocab     []wordEntry
}

type wordEntry struct {
	word  string
	freq  int
	score float32
}

// Train trains a Word model from input data.
func (t *wordTrainer) Train(ctx context.Context, config *Config) (model.Model, error) {
	t.config = config
	t.wordFreq = make(map[string]int)

	// Load training data
	sentences, err := t.loadTrainingData()
	if err != nil {
		return nil, fmt.Errorf("failed to load training data: %w", err)
	}

	if len(sentences) == 0 {
		return nil, fmt.Errorf("no training data provided")
	}

	// Count word frequencies
	t.countWords(sentences)

	// Build vocabulary
	t.buildVocabulary()

	// Build model
	return t.buildModel()
}

// loadTrainingData loads and preprocesses training sentences.
func (t *wordTrainer) loadTrainingData() ([]string, error) {
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

// countWords counts word frequencies in training data.
func (t *wordTrainer) countWords(sentences []string) {
	for _, sentence := range sentences {
		words := t.splitWords(sentence)
		for _, word := range words {
			t.wordFreq[word]++
		}
	}
}

// splitWords splits text into words.
func (t *wordTrainer) splitWords(text string) []string {
	var words []string
	var currentWord strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if currentWord.Len() > 0 {
				words = append(words, currentWord.String())
				currentWord.Reset()
			}
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

// buildVocabulary builds the vocabulary from word frequencies.
func (t *wordTrainer) buildVocabulary() {
	// Convert to slice
	t.vocab = make([]wordEntry, 0, len(t.wordFreq))
	for word, freq := range t.wordFreq {
		t.vocab = append(t.vocab, wordEntry{
			word: word,
			freq: freq,
		})
	}

	// Sort by frequency (descending)
	sort.Slice(t.vocab, func(i, j int) bool {
		return t.vocab[i].freq > t.vocab[j].freq
	})

	// Keep top N words
	if len(t.vocab) > t.config.VocabSize {
		t.vocab = t.vocab[:t.config.VocabSize]
	}

	// Assign scores
	for i := range t.vocab {
		t.vocab[i].score = float32(-i)
	}
}

// buildModel constructs the final Word model.
func (t *wordTrainer) buildModel() (model.Model, error) {
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
			Piece: strPtr(entry.word),
			Score: floatPtr(entry.score),
			Type:  &normalType,
		})
	}

	// Build trainer spec
	modelType := pb.TrainerSpec_WORD
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
		AddDummyPrefix:         boolPtr(false), // Word model doesn't use dummy prefix
		RemoveExtraWhitespaces: boolPtr(true),
		EscapeWhitespaces:      boolPtr(false),
	}

	// Create ModelProto
	modelProto := &pb.ModelProto{
		Pieces:         pieces,
		TrainerSpec:    trainerSpec,
		NormalizerSpec: normalizerSpec,
	}

	// Build and return Word model
	return model.NewWordModel(modelProto)
}
