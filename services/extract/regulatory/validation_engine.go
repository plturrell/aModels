package regulatory

import (
	"context"
	"fmt"
	"log"
	"strings"
)

// ValidationEngine validates regulatory specifications.
type ValidationEngine struct {
	schemaRepo *RegulatorySchemaRepository
	logger     *log.Logger
}

// ValidationResult represents the result of validation.
type ValidationResult struct {
	Valid           bool
	Errors          []ValidationError
	Warnings        []ValidationWarning
	Completeness    float64
	FieldCount      int
	RequiredFields  int
	ValidatedFields int
}

// ValidationError represents a validation error.
type ValidationError struct {
	FieldID      string
	RuleID       string
	ErrorType    string
	Message      string
	Severity     string
	RegulatoryRef string
}

// ValidationWarning represents a validation warning.
type ValidationWarning struct {
	FieldID      string
	RuleID       string
	WarningType  string
	Message      string
	RegulatoryRef string
}

// NewValidationEngine creates a new validation engine.
func NewValidationEngine(schemaRepo *RegulatorySchemaRepository, logger *log.Logger) *ValidationEngine {
	return &ValidationEngine{
		schemaRepo: schemaRepo,
		logger:     logger,
	}
}

// ValidateSpec validates a regulatory specification.
func (ve *ValidationEngine) ValidateSpec(ctx context.Context, spec *RegulatorySpec) (*ValidationResult, error) {
	result := &ValidationResult{
		Valid:    true,
		Errors:   []ValidationError{},
		Warnings: []ValidationWarning{},
	}

	// Get reference schema for this regulatory type
	referenceSchema, err := ve.schemaRepo.GetReferenceSchema(ctx, spec.RegulatoryType, spec.DocumentVersion)
	if err != nil {
		// If no reference schema, use basic validation
		ve.validateBasic(spec, result)
		return result, nil
	}

	// Validate against reference schema
	ve.validateAgainstSchema(spec, referenceSchema, result)

	// Validate completeness
	ve.validateCompleteness(spec, referenceSchema, result)

	// Validate field definitions
	ve.validateFieldDefinitions(spec, result)

	// Validate validation rules
	ve.validateValidationRules(spec, result)

	result.Valid = len(result.Errors) == 0

	return result, nil
}

// validateBasic performs basic validation without reference schema.
func (ve *ValidationEngine) validateBasic(spec *RegulatorySpec, result *ValidationResult) {
	// Check for required components
	if spec.ReportStructure.ReportName == "" {
		result.Errors = append(result.Errors, ValidationError{
			ErrorType: "missing_component",
			Message:   "Report name is required",
			Severity:  "error",
		})
	}

	if len(spec.FieldDefinitions) == 0 {
		result.Warnings = append(result.Warnings, ValidationWarning{
			WarningType: "incomplete",
			Message:     "No field definitions extracted",
		})
	}

	result.FieldCount = spec.ReportStructure.TotalFields
	result.RequiredFields = spec.ReportStructure.RequiredFields
}

// validateAgainstSchema validates spec against reference schema.
func (ve *ValidationEngine) validateAgainstSchema(spec *RegulatorySpec, referenceSchema *RegulatorySpec, result *ValidationResult) {
	// Check report structure matches
	if spec.ReportStructure.ReportID != referenceSchema.ReportStructure.ReportID {
		result.Warnings = append(result.Warnings, ValidationWarning{
			WarningType: "schema_mismatch",
			Message:     fmt.Sprintf("Report ID mismatch: expected %s, got %s", referenceSchema.ReportStructure.ReportID, spec.ReportStructure.ReportID),
		})
	}

	// Check required fields exist
	requiredFieldMap := make(map[string]bool)
	for _, field := range referenceSchema.FieldDefinitions {
		if field.Required {
			requiredFieldMap[field.FieldID] = true
		}
	}

	for _, field := range spec.FieldDefinitions {
		if requiredFieldMap[field.FieldID] {
			result.ValidatedFields++
			delete(requiredFieldMap, field.FieldID)
		}
	}

	// Report missing required fields
	for fieldID := range requiredFieldMap {
		result.Errors = append(result.Errors, ValidationError{
			FieldID:   fieldID,
			ErrorType: "missing_required_field",
			Message:   fmt.Sprintf("Required field %s is missing", fieldID),
			Severity:  "error",
		})
	}
}

// validateCompleteness validates completeness of the specification.
func (ve *ValidationEngine) validateCompleteness(spec *RegulatorySpec, referenceSchema *RegulatorySpec, result *ValidationResult) {
	totalFields := len(referenceSchema.FieldDefinitions)
	extractedFields := len(spec.FieldDefinitions)

	if totalFields > 0 {
		result.Completeness = float64(extractedFields) / float64(totalFields) * 100.0
	} else {
		result.Completeness = 100.0
	}

	if result.Completeness < 80.0 {
		result.Warnings = append(result.Warnings, ValidationWarning{
			WarningType: "low_completeness",
			Message:     fmt.Sprintf("Completeness is %.1f%%, below recommended 80%%", result.Completeness),
		})
	}

	result.FieldCount = extractedFields
	result.RequiredFields = spec.ReportStructure.RequiredFields
}

// validateFieldDefinitions validates field definitions.
func (ve *ValidationEngine) validateFieldDefinitions(spec *RegulatorySpec, result *ValidationResult) {
	fieldIDMap := make(map[string]bool)

	for _, field := range spec.FieldDefinitions {
		// Check for duplicate field IDs
		if fieldIDMap[field.FieldID] {
			result.Errors = append(result.Errors, ValidationError{
				FieldID:   field.FieldID,
				ErrorType: "duplicate_field",
				Message:   fmt.Sprintf("Duplicate field ID: %s", field.FieldID),
				Severity:  "error",
			})
			continue
		}
		fieldIDMap[field.FieldID] = true

		// Validate field type
		validTypes := []string{"text", "number", "date", "currency", "percentage", "boolean"}
		validType := false
		for _, validT := range validTypes {
			if field.FieldType == validT {
				validType = true
				break
			}
		}

		if !validType && field.FieldType != "" {
			result.Warnings = append(result.Warnings, ValidationWarning{
				FieldID:     field.FieldID,
				WarningType: "invalid_field_type",
				Message:     fmt.Sprintf("Field type '%s' may not be standard", field.FieldType),
			})
		}

		// Check for required field without name
		if field.Required && field.FieldName == "" {
			result.Errors = append(result.Errors, ValidationError{
				FieldID:   field.FieldID,
				ErrorType: "missing_field_name",
				Message:   fmt.Sprintf("Required field %s is missing a name", field.FieldID),
				Severity:  "error",
			})
		}
	}
}

// validateValidationRules validates validation rules.
func (ve *ValidationEngine) validateValidationRules(spec *RegulatorySpec, result *ValidationResult) {
	for _, rule := range spec.ValidationRules {
		// Check rule has expression
		if rule.Expression == "" && rule.RuleType != "manual" {
			result.Warnings = append(result.Warnings, ValidationWarning{
				RuleID:      rule.RuleID,
				WarningType: "missing_expression",
				Message:     fmt.Sprintf("Validation rule %s is missing an expression", rule.RuleID),
			})
		}

		// Validate rule type
		validTypes := []string{"range", "format", "calculation", "cross_field", "completeness", "manual"}
		validType := false
		for _, validT := range validTypes {
			if rule.RuleType == validT {
				validType = true
				break
			}
		}

		if !validType {
			result.Warnings = append(result.Warnings, ValidationWarning{
				RuleID:      rule.RuleID,
				WarningType: "unknown_rule_type",
				Message:     fmt.Sprintf("Unknown rule type: %s", rule.RuleType),
			})
		}
	}
}

// ValidateDataAgainstSpec validates data against a regulatory specification.
func (ve *ValidationEngine) ValidateDataAgainstSpec(ctx context.Context, spec *RegulatorySpec, data map[string]interface{}) (*ValidationResult, error) {
	result := &ValidationResult{
		Valid:    true,
		Errors:   []ValidationError{},
		Warnings: []ValidationWarning{},
	}

	// Validate required fields are present
	for _, field := range spec.FieldDefinitions {
		if field.Required {
			if _, exists := data[field.FieldID]; !exists {
				result.Errors = append(result.Errors, ValidationError{
					FieldID:   field.FieldID,
					ErrorType: "missing_required_field",
					Message:   fmt.Sprintf("Required field %s is missing", field.FieldID),
					Severity:  "error",
				})
			}
		}
	}

	// Validate field types
	for fieldID, value := range data {
		field := ve.findFieldDefinition(spec, fieldID)
		if field != nil {
			if !ve.validateFieldType(field, value) {
				result.Errors = append(result.Errors, ValidationError{
					FieldID:   fieldID,
					ErrorType: "type_mismatch",
					Message:   fmt.Sprintf("Field %s type mismatch: expected %s", fieldID, field.FieldType),
					Severity:  "error",
				})
			}
		}
	}

	// Validate validation rules
	for _, rule := range spec.ValidationRules {
		if err := ve.validateRule(rule, data); err != nil {
			result.Errors = append(result.Errors, ValidationError{
				RuleID:    rule.RuleID,
				ErrorType: "validation_failed",
				Message:   err.Error(),
				Severity:  rule.Severity,
			})
		}
	}

	result.Valid = len(result.Errors) == 0
	return result, nil
}

// findFieldDefinition finds a field definition by ID.
func (ve *ValidationEngine) findFieldDefinition(spec *RegulatorySpec, fieldID string) *FieldDefinition {
	for _, field := range spec.FieldDefinitions {
		if field.FieldID == fieldID {
			return &field
		}
	}
	return nil
}

// validateFieldType validates that a value matches the field type.
func (ve *ValidationEngine) validateFieldType(field *FieldDefinition, value interface{}) bool {
	switch field.FieldType {
	case "text":
		_, ok := value.(string)
		return ok
	case "number":
		switch value.(type) {
		case int, int64, float32, float64:
			return true
		}
		return false
	case "date":
		_, ok := value.(string)
		return ok // Simplified - would validate date format
	case "currency":
		switch value.(type) {
		case float32, float64:
			return true
		}
		return false
	case "percentage":
		switch value.(type) {
		case float32, float64:
			return true
		}
		return false
	case "boolean":
		_, ok := value.(bool)
		return ok
	default:
		return true // Unknown type, allow
	}
}

// validateRule validates a rule against data.
func (ve *ValidationEngine) validateRule(rule ValidationRule, data map[string]interface{}) error {
	// Simplified validation - in production would evaluate expressions
	if rule.Expression == "" {
		return nil
	}

	// Basic expression validation
	if strings.Contains(rule.Expression, "required") {
		// Check required fields
		return nil // Simplified
	}

	return nil
}

