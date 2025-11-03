package trainer

import (
	"context"
	"fmt"
	"sort"
	"unicode/utf8"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

// bpeTrainer implements the Byte Pair Encoding trainer.
type bpeTrainer struct {
	config *Config
	vocab  map[string]int // piece -> frequency
	merges []bpeMerge     // ordered list of merges
}

// bpeMerge represents a BPE merge operation.
type bpeMerge struct {
	left  string
	right string
	freq  int
}

// Train trains a BPE model from input data.
func (t *bpeTrainer) Train(ctx context.Context, config *Config) (model.Model, error) {
	t.config = config
	t.vocab = make(map[string]int)
	t.merges = make([]bpeMerge, 0)

	// Load training data
	sentences, err := t.loadTrainingData()
	if err != nil {
		return nil, fmt.Errorf("failed to load training data: %w", err)
	}

	if len(sentences) == 0 {
		return nil, fmt.Errorf("no training data provided")
	}

	// Initialize with character-level vocabulary
	t.initializeVocabulary(sentences)

	// Learn BPE merges
	targetMerges := t.config.VocabSize - len(t.vocab)
	for i := 0; i < targetMerges; i++ {
		if !t.learnNextMerge() {
			break // No more merges possible
		}
	}

	// Build final model
	return t.buildModel()
}

// loadTrainingData loads and preprocesses training sentences.
func (t *bpeTrainer) loadTrainingData() ([]string, error) {
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

// initializeVocabulary creates character-level initial vocabulary.
func (t *bpeTrainer) initializeVocabulary(sentences []string) {
	for _, sentence := range sentences {
		for _, r := range sentence {
			char := string(r)
			t.vocab[char]++
		}
	}
}

// learnNextMerge finds and applies the most frequent pair merge.
func (t *bpeTrainer) learnNextMerge() bool {
	// Count all adjacent pairs
	pairCounts := make(map[string]int)
	
	// Re-tokenize all training data with current vocab
	for _, sentence := range t.config.InputData {
		tokens := t.tokenize(sentence)
		for i := 0; i < len(tokens)-1; i++ {
			pair := tokens[i] + " " + tokens[i+1]
			pairCounts[pair]++
		}
	}

	// Find most frequent pair
	var bestPair string
	bestCount := 0
	for pair, count := range pairCounts {
		if count > bestCount {
			bestCount = count
			bestPair = pair
		}
	}

	if bestCount == 0 {
		return false
	}

	// Parse the pair
	var left, right string
	for i := 0; i < len(bestPair); i++ {
		if bestPair[i] == ' ' {
			left = bestPair[:i]
			right = bestPair[i+1:]
			break
		}
	}

	// Add merge
	merged := left + right
	t.merges = append(t.merges, bpeMerge{
		left:  left,
		right: right,
		freq:  bestCount,
	})
	t.vocab[merged] = bestCount

	return true
}

// tokenize applies current BPE rules to tokenize text.
func (t *bpeTrainer) tokenize(text string) []string {
	// Start with characters
	tokens := make([]string, 0)
	for _, r := range text {
		tokens = append(tokens, string(r))
	}

	// Apply merges in order
	for _, merge := range t.merges {
		newTokens := make([]string, 0)
		i := 0
		for i < len(tokens) {
			if i < len(tokens)-1 && tokens[i] == merge.left && tokens[i+1] == merge.right {
				newTokens = append(newTokens, merge.left+merge.right)
				i += 2
			} else {
				newTokens = append(newTokens, tokens[i])
				i++
			}
		}
		tokens = newTokens
	}

	return tokens
}

// buildModel constructs the final BPE model.
func (t *bpeTrainer) buildModel() (model.Model, error) {
	// Sort vocabulary by frequency (descending)
	type vocabEntry struct {
		piece string
		freq  int
	}
	
	vocabList := make([]vocabEntry, 0, len(t.vocab))
	for piece, freq := range t.vocab {
		vocabList = append(vocabList, vocabEntry{piece: piece, freq: freq})
	}
	
	sort.Slice(vocabList, func(i, j int) bool {
		return vocabList[i].freq > vocabList[j].freq
	})

	// Build pieces
	pieces := make([]*pb.ModelProto_SentencePiece, 0, len(vocabList)+3)
	
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
	
	// Add vocabulary pieces
	for i, entry := range vocabList {
		if i >= t.config.VocabSize {
			break
		}
		pieces = append(pieces, &pb.ModelProto_SentencePiece{
			Piece: strPtr(entry.piece),
			Score: floatPtr(float32(-i)), // Score based on order
			Type:  &normalType,
		})
	}
	
	// Build trainer spec
	modelType := pb.TrainerSpec_BPE
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
		AddDummyPrefix:         boolPtr(true),
		RemoveExtraWhitespaces: boolPtr(true),
		EscapeWhitespaces:      boolPtr(true),
	}
	
	// Create ModelProto
	modelProto := &pb.ModelProto{
		Pieces:         pieces,
		TrainerSpec:    trainerSpec,
		NormalizerSpec: normalizerSpec,
	}
	
	// Build and return BPE model
	return model.NewBPEModel(modelProto)
}
