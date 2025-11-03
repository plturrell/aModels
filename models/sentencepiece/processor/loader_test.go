package processor

import (
	"os"
	"testing"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
	"google.golang.org/protobuf/proto"
)

func TestLoadModel_InvalidFile(t *testing.T) {
	p := New()
	err := p.LoadModel("/nonexistent/model.model")
	if err == nil {
		t.Error("Expected error for nonexistent file, got nil")
	}
}

func TestLoadModel_InvalidProtobuf(t *testing.T) {
	// Create a temporary file with invalid protobuf data
	tmpfile := t.TempDir() + "/invalid.model"
	data := []byte("not a valid protobuf")
	
	if err := writeFile(tmpfile, data); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	p := New()
	err := p.LoadModel(tmpfile)
	if err == nil {
		t.Error("Expected error for invalid protobuf, got nil")
	}
}

func TestLoadModel_MissingTrainerSpec(t *testing.T) {
	// Create a model proto without trainer spec
	modelProto := &pb.ModelProto{
		// No trainer spec
	}
	
	data, err := proto.Marshal(modelProto)
	if err != nil {
		t.Fatalf("Failed to marshal proto: %v", err)
	}

	tmpfile := t.TempDir() + "/no_trainer.model"
	if err := writeFile(tmpfile, data); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	p := New()
	err = p.LoadModel(tmpfile)
	if err == nil {
		t.Error("Expected error for missing trainer spec, got nil")
	}
	if err != nil && err.Error() != "model missing trainer spec" {
		t.Errorf("Expected 'model missing trainer spec' error, got: %v", err)
	}
}

func TestLoadModel_UnsupportedModelType(t *testing.T) {
	// Create a model proto with unsupported model type
	modelType := pb.TrainerSpec_UNIGRAM
	modelProto := &pb.ModelProto{
		TrainerSpec: &pb.TrainerSpec{
			ModelType: &modelType,
		},
	}
	
	data, err := proto.Marshal(modelProto)
	if err != nil {
		t.Fatalf("Failed to marshal proto: %v", err)
	}

	tmpfile := t.TempDir() + "/unsupported.model"
	if err := writeFile(tmpfile, data); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	p := New()
	err = p.LoadModel(tmpfile)
	// Should get "not yet implemented" error since we haven't implemented model loading
	if err == nil {
		t.Error("Expected error for unimplemented model type, got nil")
	}
}

// Helper function to write test files
func writeFile(path string, data []byte) error {
	return os.WriteFile(path, data, 0644)
}
