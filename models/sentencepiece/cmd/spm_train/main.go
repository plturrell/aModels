package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"google.golang.org/protobuf/proto"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/trainer"
)

func main() {
	// Command-line flags
	var (
		inputFiles        = flag.String("input", "", "Comma-separated input files")
		modelPrefix       = flag.String("model_prefix", "m", "Output model prefix")
		vocabSize         = flag.Int("vocab_size", 8000, "Vocabulary size")
		modelType         = flag.String("model_type", "unigram", "Model type (unigram, bpe, word, char)")
		characterCoverage = flag.Float64("character_coverage", 0.9995, "Character coverage")
		maxSentenceLength = flag.Int("max_sentence_length", 4192, "Maximum sentence length")
	)

	flag.Parse()

	if *inputFiles == "" {
		fmt.Fprintf(os.Stderr, "Error: --input is required\n")
		flag.Usage()
		os.Exit(1)
	}

	// Determine model type
	var mType model.Type
	switch *modelType {
	case "unigram":
		mType = model.TypeUnigram
	case "bpe":
		mType = model.TypeBPE
	case "word":
		mType = model.TypeWord
	case "char":
		mType = model.TypeChar
	default:
		fmt.Fprintf(os.Stderr, "Error: unknown model type: %s\n", *modelType)
		os.Exit(1)
	}

	// Read input files
	inputData, err := readInputFiles(*inputFiles)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input files: %v\n", err)
		os.Exit(1)
	}

	// Create training config
	config := &trainer.Config{
		InputData:         inputData,
		ModelType:         mType,
		VocabSize:         *vocabSize,
		CharacterCoverage: *characterCoverage,
		MaxSentenceLength: *maxSentenceLength,
		ModelPrefix:       *modelPrefix,
	}

	// Create trainer
	t, err := trainer.New(mType)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating trainer: %v\n", err)
		os.Exit(1)
	}

	// Train model
	fmt.Printf("Training %s model with vocab_size=%d...\n", *modelType, *vocabSize)
	trainedModel, err := t.Train(context.Background(), config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error training model: %v\n", err)
		os.Exit(1)
	}

	// Save model (need to extract ModelProto from trained model)
	// For now, create a simple model file
	modelFile := *modelPrefix + ".model"
	vocabFile := *modelPrefix + ".vocab"

	fmt.Printf("Saving model to %s...\n", modelFile)
	if err := saveModel(trainedModel, modelFile); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving model: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Saving vocabulary to %s...\n", vocabFile)
	if err := saveVocab(trainedModel, vocabFile); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving vocabulary: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Training complete!")
}

func readInputFiles(files string) ([]string, error) {
	// For simplicity, read from a single file
	data, err := os.ReadFile(files)
	if err != nil {
		return nil, err
	}

	// Split by newlines
	lines := []string{}
	current := ""
	for _, b := range data {
		if b == '\n' {
			if len(current) > 0 {
				lines = append(lines, current)
				current = ""
			}
		} else {
			current += string(b)
		}
	}
	if len(current) > 0 {
		lines = append(lines, current)
	}

	return lines, nil
}

func saveModel(m model.Model, filename string) error {
	// Create a basic ModelProto
	// This is a simplified version - full implementation would extract from model
	pieces := make([]*pb.ModelProto_SentencePiece, m.GetPieceSize())
	
	normalType := pb.ModelProto_SentencePiece_NORMAL
	for i := 0; i < m.GetPieceSize(); i++ {
		piece, _ := m.GetPiece(i)
		score, _ := m.GetScore(i)
		
		pieces[i] = &pb.ModelProto_SentencePiece{
			Piece: &piece,
			Score: &score,
			Type:  &normalType,
		}
	}

	modelProto := &pb.ModelProto{
		Pieces: pieces,
	}

	data, err := proto.Marshal(modelProto)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

func saveVocab(m model.Model, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	for i := 0; i < m.GetPieceSize(); i++ {
		piece, _ := m.GetPiece(i)
		score, _ := m.GetScore(i)
		fmt.Fprintf(f, "%s\t%f\n", piece, score)
	}

	return nil
}
