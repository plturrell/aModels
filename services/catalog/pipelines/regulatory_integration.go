package pipelines

import (
	"context"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/extract/regulatory"
)

// RegulatoryPipelineIntegration integrates regulatory specs with semantic pipelines.
type RegulatoryPipelineIntegration struct {
	regSystem *regulatory.RegulatorySpecSystem
	logger    *log.Logger
}

// NewRegulatoryPipelineIntegration creates a new regulatory pipeline integration.
func NewRegulatoryPipelineIntegration(regSystem *regulatory.RegulatorySpecSystem, logger *log.Logger) *RegulatoryPipelineIntegration {
	return &RegulatoryPipelineIntegration{
		regSystem: regSystem,
		logger:    logger,
	}
}

// ValidatePipelineAgainstRegulatorySpec validates a pipeline against a regulatory specification.
func (rpi *RegulatoryPipelineIntegration) ValidatePipelineAgainstRegulatorySpec(ctx context.Context, pipeline *SemanticPipeline, specID string) (*RegulatoryValidationResult, error) {
	if rpi.regSystem == nil {
		return nil, fmt.Errorf("regulatory system not initialized")
	}

	// Get regulatory spec
	spec, err := rpi.regSystem.GetSchema(ctx, specID)
	if err != nil {
		return nil, fmt.Errorf("failed to get regulatory spec: %w", err)
	}

	// Validate pipeline fields against spec
	result := &RegulatoryValidationResult{
		SpecID:     specID,
		Valid:      true,
		Violations: []string{},
	}

	// Check required fields from spec are present in pipeline
	for _, field := range spec.FieldDefinitions {
		if field.Required {
			found := false
			for _, step := range pipeline.Steps {
				for _, output := range step.Outputs {
					if output.Name == field.FieldName {
						found = true
						break
					}
				}
				if found {
					break
				}
			}
			if !found {
				result.Valid = false
				result.Violations = append(result.Violations, fmt.Sprintf("Required field %s not found in pipeline", field.FieldName))
			}
		}
	}

	if rpi.logger != nil {
		if result.Valid {
			rpi.logger.Printf("Pipeline validated against regulatory spec %s", specID)
		} else {
			rpi.logger.Printf("Pipeline validation failed against regulatory spec %s: %v", specID, result.Violations)
		}
	}

	return result, nil
}

// MapSpecToPipeline creates a pipeline definition from a regulatory specification.
func (rpi *RegulatoryPipelineIntegration) MapSpecToPipeline(ctx context.Context, specID string) (*SemanticPipeline, error) {
	if rpi.regSystem == nil {
		return nil, fmt.Errorf("regulatory system not initialized")
	}

	// Get regulatory spec
	spec, err := rpi.regSystem.GetSchema(ctx, specID)
	if err != nil {
		return nil, fmt.Errorf("failed to get regulatory spec: %w", err)
	}

	// Create pipeline from spec
	pipeline := &SemanticPipeline{
		ID:          fmt.Sprintf("regulatory-%s", specID),
		Name:        fmt.Sprintf("Pipeline for %s", spec.ReportStructure.ReportName),
		Description: fmt.Sprintf("Pipeline generated from regulatory spec %s", specID),
		Steps:       []PipelineStep{},
	}

	// Create steps for each field
	for _, field := range spec.FieldDefinitions {
		step := PipelineStep{
			ID:          fmt.Sprintf("step-%s", field.FieldID),
			Name:        fmt.Sprintf("Extract %s", field.FieldName),
			Type:        "extract",
			Description: field.Description,
			Outputs: []PipelineOutput{
				{
					Name:        field.FieldName,
					Type:        field.FieldType,
					Required:    field.Required,
					Description: field.Description,
				},
			},
		}
		pipeline.Steps = append(pipeline.Steps, step)
	}

	if rpi.logger != nil {
		rpi.logger.Printf("Created pipeline from regulatory spec %s with %d steps", specID, len(pipeline.Steps))
	}

	return pipeline, nil
}

// RegulatoryValidationResult represents the result of regulatory validation.
type RegulatoryValidationResult struct {
	SpecID     string
	Valid      bool
	Violations []string
}

