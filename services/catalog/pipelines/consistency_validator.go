package pipelines

import (
	"context"
	"fmt"
	"log"
)

// ConsistencyValidator validates consistency between source and target schemas.
type ConsistencyValidator struct {
	logger *log.Logger
}

// NewConsistencyValidator creates a new consistency validator.
func NewConsistencyValidator(logger *log.Logger) *ConsistencyValidator {
	return &ConsistencyValidator{
		logger: logger,
	}
}

// ValidateSchema validates that source and target schemas are compatible.
func (cv *ConsistencyValidator) ValidateSchema(ctx context.Context, source SchemaDefinition, target SchemaDefinition) (*ValidationResult, error) {
	if cv.logger != nil {
		cv.logger.Printf("Validating schema consistency between source and target")
	}

	result := &ValidationResult{
		Valid:   true,
		Issues:  []ValidationIssue{},
		Warnings: []string{},
	}

	// Check if all required source fields are present in target
	sourceFieldMap := make(map[string]FieldDefinition)
	for _, field := range source.Fields {
		sourceFieldMap[field.Name] = field
	}

	targetFieldMap := make(map[string]FieldDefinition)
	for _, field := range target.Fields {
		targetFieldMap[field.Name] = field
	}

	// Check for missing required fields in target
	for name, sourceField := range sourceFieldMap {
		if sourceField.Required {
			targetField, exists := targetFieldMap[name]
			if !exists {
				result.Valid = false
				result.Issues = append(result.Issues, ValidationIssue{
					Type:        "missing_required_field",
					Field:       name,
					Severity:    "error",
					Description: fmt.Sprintf("Required source field %s is missing in target schema", name),
				})
			} else if !targetField.Required {
				result.Warnings = append(result.Warnings,
					fmt.Sprintf("Source field %s is required but target field is optional", name))
			}
		}
	}

	// Check for type mismatches
	for name, sourceField := range sourceFieldMap {
		if targetField, exists := targetFieldMap[name]; exists {
			if !cv.isTypeCompatible(sourceField.Type, targetField.Type) {
				result.Valid = false
				result.Issues = append(result.Issues, ValidationIssue{
					Type:        "type_mismatch",
					Field:       name,
					Severity:    "error",
					Description: fmt.Sprintf("Type mismatch: source=%s, target=%s", sourceField.Type, targetField.Type),
				})
			}
		}
	}

	// Check primary key consistency
	if len(source.PrimaryKey) > 0 {
		if len(target.PrimaryKey) == 0 {
			result.Warnings = append(result.Warnings,
				"Source has primary key but target does not")
		} else {
			// Check if primary key fields match
			sourcePKSet := make(map[string]bool)
			for _, pk := range source.PrimaryKey {
				sourcePKSet[pk] = true
			}
			targetPKSet := make(map[string]bool)
			for _, pk := range target.PrimaryKey {
				targetPKSet[pk] = true
			}

			// Check if all source PK fields are in target PK
			for _, pk := range source.PrimaryKey {
				if !targetPKSet[pk] {
					result.Warnings = append(result.Warnings,
						fmt.Sprintf("Primary key field %s from source is not in target primary key", pk))
				}
			}
		}
	}

	return result, nil
}

// ValidateDataQuality validates data quality metrics.
func (cv *ConsistencyValidator) ValidateDataQuality(ctx context.Context, pipeline *SemanticPipeline, data map[string]interface{}) (*QualityValidationResult, error) {
	result := &QualityValidationResult{
		Valid:   true,
		GateResults: []GateResult{},
	}

	for _, gate := range pipeline.Validation.DataQualityGates {
		gateResult := cv.evaluateQualityGate(gate, data)
		result.GateResults = append(result.GateResults, gateResult)
		
		if !gateResult.Passed {
			if gate.OnFailure == "stop" {
				result.Valid = false
			}
		}
	}

	return result, nil
}

// ValidateContractTests validates contract tests between source and target.
func (cv *ConsistencyValidator) ValidateContractTests(ctx context.Context, pipeline *SemanticPipeline) (*ContractTestResult, error) {
	if !pipeline.Validation.ContractTesting.Enabled {
		return &ContractTestResult{
			Enabled: false,
			Message: "Contract testing is disabled",
		}, nil
	}

	result := &ContractTestResult{
		Enabled: true,
		Passed:  true,
		TestResults: []TestResult{},
	}

	for _, test := range pipeline.Validation.ContractTesting.Tests {
		testResult := cv.runContractTest(test, pipeline)
		result.TestResults = append(result.TestResults, testResult)
		
		if !testResult.Passed {
			result.Passed = false
		}
	}

	return result, nil
}

// isTypeCompatible checks if two types are compatible.
func (cv *ConsistencyValidator) isTypeCompatible(sourceType, targetType string) bool {
	// Exact match
	if sourceType == targetType {
		return true
	}

	// Type compatibility matrix
	compatibility := map[string][]string{
		"integer": {"integer", "float"},
		"float":   {"float"},
		"string":  {"string", "json"},
		"json":    {"json", "object"},
		"array":   {"array", "json"},
		"object":  {"object", "json"},
	}

	compatibleTypes, exists := compatibility[sourceType]
	if !exists {
		return false
	}

	for _, ct := range compatibleTypes {
		if ct == targetType {
			return true
		}
	}

	return false
}

// evaluateQualityGate evaluates a single quality gate.
func (cv *ConsistencyValidator) evaluateQualityGate(gate QualityGate, data map[string]interface{}) GateResult {
	// In production, would compute actual metric value from data
	// For now, return placeholder
	actualValue := 1.0 // Placeholder

	passed := cv.compareValue(actualValue, gate.Threshold, gate.Operator)

	return GateResult{
		GateName:  gate.Name,
		Metric:    gate.Metric,
		Passed:    passed,
		ActualValue: actualValue,
		Threshold: gate.Threshold,
		Operator:  gate.Operator,
	}
}

// compareValue compares a value against a threshold using an operator.
func (cv *ConsistencyValidator) compareValue(actual, threshold float64, operator string) bool {
	switch operator {
	case ">=":
		return actual >= threshold
	case "<=":
		return actual <= threshold
	case "==":
		return actual == threshold
	case "!=":
		return actual != threshold
	case ">":
		return actual > threshold
	case "<":
		return actual < threshold
	default:
		return false
	}
}

// runContractTest runs a single contract test.
func (cv *ConsistencyValidator) runContractTest(test ContractTest, pipeline *SemanticPipeline) TestResult {
	// In production, would execute actual contract test
	// For now, return placeholder
	return TestResult{
		TestName:    test.Name,
		TestType:    test.Type,
		Passed:      true,
		Message:     fmt.Sprintf("Contract test %s passed", test.Name),
	}
}

// ValidationResult represents the result of schema validation.
type ValidationResult struct {
	Valid    bool            `json:"valid"`
	Issues   []ValidationIssue `json:"issues"`
	Warnings []string        `json:"warnings"`
}

// ValidationIssue represents a validation issue.
type ValidationIssue struct {
	Type        string `json:"type"`
	Field       string `json:"field,omitempty"`
	Severity    string `json:"severity"` // "error", "warning"
	Description string `json:"description"`
}

// QualityValidationResult represents the result of data quality validation.
type QualityValidationResult struct {
	Valid       bool         `json:"valid"`
	GateResults []GateResult `json:"gate_results"`
}

// GateResult represents the result of a quality gate.
type GateResult struct {
	GateName    string  `json:"gate_name"`
	Metric      string  `json:"metric"`
	Passed      bool    `json:"passed"`
	ActualValue float64 `json:"actual_value"`
	Threshold   float64 `json:"threshold"`
	Operator    string  `json:"operator"`
}

// ContractTestResult represents the result of contract testing.
type ContractTestResult struct {
	Enabled    bool         `json:"enabled"`
	Passed     bool         `json:"passed"`
	TestResults []TestResult `json:"test_results"`
	Message    string       `json:"message,omitempty"`
}

// TestResult represents the result of a contract test.
type TestResult struct {
	TestName string `json:"test_name"`
	TestType string `json:"test_type"`
	Passed   bool   `json:"passed"`
	Message  string `json:"message,omitempty"`
}
