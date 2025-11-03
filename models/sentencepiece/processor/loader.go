package processor

import (
	"fmt"
	"os"

	"google.golang.org/protobuf/proto"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/normalizer"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/model"
)

// LoadModel loads a SentencePiece model from a protobuf file.
func (p *Processor) LoadModel(modelPath string) error {
	// Read the model file
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("failed to read model file: %w", err)
	}

	// Parse the protobuf
	var modelProto pb.ModelProto
	if err := proto.Unmarshal(data, &modelProto); err != nil {
		return fmt.Errorf("failed to unmarshal model proto: %w", err)
	}

	return p.loadFromProto(&modelProto)
}

// loadFromProto loads a model from a ModelProto.
func (p *Processor) loadFromProto(modelProto *pb.ModelProto) error {

	// Extract trainer spec
	if modelProto.TrainerSpec == nil {
		return fmt.Errorf("model missing trainer spec")
	}

	// Determine model type
	var modelType model.Type
	switch modelProto.TrainerSpec.GetModelType() {
	case pb.TrainerSpec_UNIGRAM:
		modelType = model.TypeUnigram
	case pb.TrainerSpec_BPE:
		modelType = model.TypeBPE
	case pb.TrainerSpec_WORD:
		modelType = model.TypeWord
	case pb.TrainerSpec_CHAR:
		modelType = model.TypeChar
	default:
		return fmt.Errorf("unknown model type: %v", modelProto.TrainerSpec.GetModelType())
	}

	// Create the appropriate model implementation
	var err error
	switch modelType {
	case model.TypeUnigram:
		p.model, err = loadUnigramModel(modelProto)
		p.modelType = "UNIGRAM"
	case model.TypeBPE:
		p.model, err = loadBPEModel(modelProto)
		p.modelType = "BPE"
	case model.TypeWord:
		p.model, err = loadWordModel(modelProto)
		p.modelType = "WORD"
	case model.TypeChar:
		p.model, err = loadCharModel(modelProto)
		p.modelType = "CHAR"
	default:
		return fmt.Errorf("unsupported model type: %v", modelType)
	}

	if err != nil {
		return fmt.Errorf("failed to load %s model: %w", modelType, err)
	}

	// Initialize normalizer
	if modelProto.NormalizerSpec != nil {
		normConfig := &normalizer.Config{
			AddDummyPrefix:         modelProto.NormalizerSpec.GetAddDummyPrefix(),
			RemoveExtraWhitespaces: modelProto.NormalizerSpec.GetRemoveExtraWhitespaces(),
			EscapeWhitespaces:      modelProto.NormalizerSpec.GetEscapeWhitespaces(),
		}

		// Use precompiled charsmap if available
		if len(modelProto.NormalizerSpec.GetPrecompiledCharsmap()) > 0 {
			p.normalizer = normalizer.NewFromPrecompiledCharsmap(
				modelProto.NormalizerSpec.GetPrecompiledCharsmap(),
				normConfig,
			)
		} else {
			p.normalizer = normalizer.New(normConfig)
		}
	} else {
		// Use default normalizer
		p.normalizer = normalizer.New(nil)
	}

	return nil
}

// LoadModelFromBytes loads a SentencePiece model from protobuf bytes.
func (p *Processor) LoadModelFromBytes(data []byte) error {
	// Parse the protobuf
	var modelProto pb.ModelProto
	if err := proto.Unmarshal(data, &modelProto); err != nil {
		return fmt.Errorf("failed to unmarshal model proto: %w", err)
	}

	// Delegate to LoadModel logic (refactor to share code)
	// TODO: Refactor to avoid duplication
	return fmt.Errorf("not yet implemented")
}

// loadUnigramModel creates a Unigram model from the protobuf.
func loadUnigramModel(modelProto *pb.ModelProto) (model.Model, error) {
	return model.NewUnigramModel(modelProto)
}

// loadBPEModel creates a BPE model from the protobuf.
func loadBPEModel(modelProto *pb.ModelProto) (model.Model, error) {
	return model.NewBPEModel(modelProto)
}

// loadWordModel creates a Word model from the protobuf.
func loadWordModel(modelProto *pb.ModelProto) (model.Model, error) {
	return model.NewWordModel(modelProto)
}

// loadCharModel creates a Char model from the protobuf.
func loadCharModel(modelProto *pb.ModelProto) (model.Model, error) {
	return model.NewCharModel(modelProto)
}
