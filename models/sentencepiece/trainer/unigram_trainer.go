package trainer

import (
	"context"
	"fmt"
	"math"
	"sort"
	"unicode/utf8"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

// unigramTrainer implements the Unigram language model trainer.
type unigramTrainer struct {
	config *Config
	// Vocabulary candidates with their frequencies
	candidates map[string]*candidatePiece
	// Final vocabulary
	vocab []vocabPiece
}

// candidatePiece represents a candidate subword piece during training.
type candidatePiece struct {
	piece     string
	freq      float64
	score     float64
}

// vocabPiece represents a final vocabulary entry.
type vocabPiece struct {
	piece string
	score float32
	freq  float64
}

// Train trains a Unigram model from input data.
func (t *unigramTrainer) Train(ctx context.Context, config *Config) (model.Model, error) {
	t.config = config
	t.candidates = make(map[string]*candidatePiece)

	// Step 1: Load and preprocess training data
	sentences, err := t.loadTrainingData()
	if err != nil {
		return nil, fmt.Errorf("failed to load training data: %w", err)
	}

	if len(sentences) == 0 {
		return nil, fmt.Errorf("no training data provided")
	}

	// Step 2: Initialize seed vocabulary with character n-grams
	if err := t.initializeSeedVocabulary(sentences); err != nil {
		return nil, fmt.Errorf("failed to initialize seed vocabulary: %w", err)
	}

	// Step 3: Run EM algorithm to optimize vocabulary
	if err := t.runEMAlgorithm(sentences); err != nil {
		return nil, fmt.Errorf("failed to run EM algorithm: %w", err)
	}

	// Step 4: Prune vocabulary to target size
	if err := t.pruneVocabulary(); err != nil {
		return nil, fmt.Errorf("failed to prune vocabulary: %w", err)
	}

	// Step 5: Build final model
	finalModel, err := t.buildModel()
	if err != nil {
		return nil, fmt.Errorf("failed to build model: %w", err)
	}

	return finalModel, nil
}

// loadTrainingData loads and preprocesses training sentences.
func (t *unigramTrainer) loadTrainingData() ([]string, error) {
	var sentences []string

	// Load from input data (strings)
	if len(t.config.InputData) > 0 {
		sentences = append(sentences, t.config.InputData...)
	}

	// TODO: Load from input files
	// if len(t.config.InputFiles) > 0 {
	//     for _, file := range t.config.InputFiles {
	//         // Read and parse file
	//     }
	// }

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

// initializeSeedVocabulary creates initial vocabulary from character n-grams.
func (t *unigramTrainer) initializeSeedVocabulary(sentences []string) error {
	// Count character frequencies
	charFreq := make(map[rune]int)
	for _, sentence := range sentences {
		for _, r := range sentence {
			charFreq[r]++
		}
	}

	// Add individual characters as seed pieces
	for r, freq := range charFreq {
		piece := string(r)
		t.candidates[piece] = &candidatePiece{
			piece: piece,
			freq:  float64(freq),
			score: 0.0,
		}
	}

	// Add frequent character bigrams and trigrams
	t.addNGrams(sentences, 2) // bigrams
	t.addNGrams(sentences, 3) // trigrams

	// Limit seed vocabulary size
	seedSize := t.config.VocabSize * 10 // 10x target size as seed
	if len(t.candidates) > seedSize {
		t.pruneCandidates(seedSize)
	}

	return nil
}

// addNGrams adds character n-grams to candidate vocabulary.
func (t *unigramTrainer) addNGrams(sentences []string, n int) {
	ngramFreq := make(map[string]int)

	for _, sentence := range sentences {
		runes := []rune(sentence)
		for i := 0; i <= len(runes)-n; i++ {
			ngram := string(runes[i : i+n])
			ngramFreq[ngram]++
		}
	}

	// Add frequent n-grams
	for ngram, freq := range ngramFreq {
		if freq >= 2 { // Minimum frequency threshold
			if _, exists := t.candidates[ngram]; !exists {
				t.candidates[ngram] = &candidatePiece{
					piece: ngram,
					freq:  float64(freq),
					score: 0.0,
				}
			}
		}
	}
}

// runEMAlgorithm runs Expectation-Maximization to optimize vocabulary.
func (t *unigramTrainer) runEMAlgorithm(sentences []string) error {
	numIterations := 10 // Default number of EM iterations
	if t.config.NumThreads > 0 {
		// Could parallelize here
	}

	for iter := 0; iter < numIterations; iter++ {
		// E-step: Compute expected counts
		if err := t.eStep(sentences); err != nil {
			return fmt.Errorf("E-step failed at iteration %d: %w", iter, err)
		}

		// M-step: Update piece scores
		t.mStep()

		// Prune low-scoring pieces periodically
		if iter > 0 && iter%2 == 0 {
			targetSize := int(float64(t.config.VocabSize) * 1.5)
			if len(t.candidates) > targetSize {
				t.pruneCandidates(targetSize)
			}
		}
	}

	return nil
}

// eStep computes expected counts for each piece (E-step of EM).
func (t *unigramTrainer) eStep(sentences []string) error {
	// Reset frequencies
	for _, cand := range t.candidates {
		cand.freq = 0.0
	}

	// For each sentence, find best segmentation and update counts
	for _, sentence := range sentences {
		segmentation := t.viterbiSegment(sentence)
		for _, piece := range segmentation {
			if cand, exists := t.candidates[piece]; exists {
				cand.freq += 1.0
			}
		}
	}

	return nil
}

// mStep updates piece scores based on frequencies (M-step of EM).
func (t *unigramTrainer) mStep() {
	// Compute total frequency
	totalFreq := 0.0
	for _, cand := range t.candidates {
		totalFreq += cand.freq
	}

	// Update scores as log probabilities
	for _, cand := range t.candidates {
		if cand.freq > 0 {
			cand.score = math.Log(cand.freq / totalFreq)
		} else {
			cand.score = -999.0 // Very low score for unseen pieces
		}
	}
}

// viterbiSegment finds the best segmentation of a sentence.
func (t *unigramTrainer) viterbiSegment(sentence string) []string {
	runes := []rune(sentence)
	n := len(runes)

	// Dynamic programming: dp[i] = best score to position i
	dp := make([]float64, n+1)
	backtrack := make([]int, n+1)

	for i := 0; i <= n; i++ {
		dp[i] = -math.MaxFloat64
		backtrack[i] = -1
	}
	dp[0] = 0.0

	// Fill DP table
	for i := 0; i < n; i++ {
		if dp[i] == -math.MaxFloat64 {
			continue
		}

		// Try all possible pieces starting at position i
		for length := 1; length <= n-i; length++ {
			piece := string(runes[i : i+length])
			if cand, exists := t.candidates[piece]; exists {
				newScore := dp[i] + cand.score
				if newScore > dp[i+length] {
					dp[i+length] = newScore
					backtrack[i+length] = i
				}
			}
		}
	}

	// Backtrack to find segmentation
	var segmentation []string
	pos := n
	for pos > 0 {
		prevPos := backtrack[pos]
		if prevPos < 0 {
			break
		}
		piece := string(runes[prevPos:pos])
		segmentation = append([]string{piece}, segmentation...)
		pos = prevPos
	}

	return segmentation
}

// pruneCandidates reduces vocabulary to target size.
func (t *unigramTrainer) pruneCandidates(targetSize int) {
	if len(t.candidates) <= targetSize {
		return
	}

	// Convert to slice for sorting
	candList := make([]*candidatePiece, 0, len(t.candidates))
	for _, cand := range t.candidates {
		candList = append(candList, cand)
	}

	// Sort by score (descending)
	sort.Slice(candList, func(i, j int) bool {
		return candList[i].score > candList[j].score
	})

	// Keep top pieces
	newCandidates := make(map[string]*candidatePiece)
	for i := 0; i < targetSize && i < len(candList); i++ {
		cand := candList[i]
		newCandidates[cand.piece] = cand
	}

	t.candidates = newCandidates
}

// pruneVocabulary prunes to final vocabulary size.
func (t *unigramTrainer) pruneVocabulary() error {
	// Convert candidates to vocab
	t.vocab = make([]vocabPiece, 0, len(t.candidates))
	for _, cand := range t.candidates {
		t.vocab = append(t.vocab, vocabPiece{
			piece: cand.piece,
			score: float32(cand.score),
			freq:  cand.freq,
		})
	}

	// Sort by score
	sort.Slice(t.vocab, func(i, j int) bool {
		return t.vocab[i].score > t.vocab[j].score
	})

	// Keep top pieces up to vocab size
	if len(t.vocab) > t.config.VocabSize {
		t.vocab = t.vocab[:t.config.VocabSize]
	}

	return nil
}

// buildModel constructs the final Unigram model.
func (t *unigramTrainer) buildModel() (model.Model, error) {
	// Build ModelProto from trained vocabulary
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
	
	// Add vocabulary pieces
	for _, vp := range t.vocab {
		pieces = append(pieces, &pb.ModelProto_SentencePiece{
			Piece: strPtr(vp.piece),
			Score: floatPtr(vp.score),
			Type:  &normalType,
		})
	}
	
	// Build trainer spec
	modelType := pb.TrainerSpec_UNIGRAM
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
	
	// Build and return Unigram model
	return model.NewUnigramModel(modelProto)
}

// Helper functions for pointer creation
func strPtr(s string) *string {
	return &s
}

func floatPtr(f float32) *float32 {
	return &f
}

func int32Ptr(i int32) *int32 {
	return &i
}

func boolPtr(b bool) *bool {
	return &b
}
