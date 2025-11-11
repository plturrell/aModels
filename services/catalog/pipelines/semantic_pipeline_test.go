package pipelines

import (
	"context"
	"log"
	"os"
	"testing"
	"time"
)

func createTestPipeline() *SemanticPipeline {
	return &SemanticPipeline{
		ID:          "test-pipeline-1",
		Name:        "Test Pipeline",
		Version:     "1.0.0",
		Source:      SourceConfig{Type: "murex", Connection: "test", Schema: SchemaDefinition{Fields: []FieldDefinition{{Name: "id", Type: "string", Required: true}}}},
		Target:      TargetConfig{Type: "knowledge_graph", Connection: "test", Schema: SchemaDefinition{Fields: []FieldDefinition{{Name: "id", Type: "string", Required: true}}}},
		Steps:       []PipelineStep{{ID: "step-1", Name: "Extract", Type: "extract", OnError: "stop"}},
		Validation:  ValidationConfig{SchemaValidation: true},
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

func TestPipelineExecutor_ValidatePipeline(t *testing.T) {
	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	executor := NewPipelineExecutor(logger)

	tests := []struct {
		name      string
		pipeline  *SemanticPipeline
		wantError bool
	}{
		{"valid", createTestPipeline(), false},
		{"missing ID", func() *SemanticPipeline { p := createTestPipeline(); p.ID = ""; return p }(), true},
		{"missing name", func() *SemanticPipeline { p := createTestPipeline(); p.Name = ""; return p }(), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := executor.validatePipeline(tt.pipeline)
			if tt.wantError && err == nil {
				t.Error("Expected error, got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestConsistencyValidator_ValidateSchema(t *testing.T) {
	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	validator := NewConsistencyValidator(logger)
	ctx := context.Background()

	source := SchemaDefinition{Fields: []FieldDefinition{{Name: "id", Type: "string", Required: true}}}
	target := SchemaDefinition{Fields: []FieldDefinition{{Name: "id", Type: "string", Required: true}}}

	result, err := validator.ValidateSchema(ctx, source, target)
	if err != nil {
		t.Fatalf("ValidateSchema() error = %v", err)
	}

	if !result.Valid {
		t.Error("ValidateSchema() Valid = false, want true for compatible schemas")
	}
}
