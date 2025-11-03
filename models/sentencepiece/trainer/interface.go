package trainer

import (
	"context"
	"errors"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

// Trainer defines the interface for training SentencePiece models.
type Trainer interface {
	// Train trains a model from input data.
	Train(ctx context.Context, config *Config) (model.Model, error)
}

// Config holds training configuration parameters.
type Config struct {
	// Input files or data
	InputFiles []string
	InputData  []string

	// Model parameters
	ModelType      model.Type
	VocabSize      int
	CharacterCoverage float64
	
	// Output
	ModelPrefix string
	
	// Training parameters
	NumThreads int
	MaxSentenceLength int
	
	// TODO: Add more configuration fields as needed
}

// New creates a new Trainer based on the model type.
func New(modelType model.Type) (Trainer, error) {
	switch modelType {
	case model.TypeUnigram:
		return &unigramTrainer{}, nil
	case model.TypeBPE:
		return &bpeTrainer{}, nil
	case model.TypeWord:
		return &wordTrainer{}, nil
	case model.TypeChar:
		return &charTrainer{}, nil
	default:
		return nil, errors.New("unsupported model type")
	}
}

// unigramTrainer is now implemented in unigram_trainer.go
// bpeTrainer is now implemented in bpe_trainer.go
// wordTrainer is now implemented in word_trainer.go
// charTrainer is now implemented in char_trainer.go
