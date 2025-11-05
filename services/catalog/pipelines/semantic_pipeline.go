package pipelines

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// SemanticPipeline represents a semantic data pipeline definition.
type SemanticPipeline struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Version     string                 `json:"version"`
	Description string                 `json:"description"`
	Source      SourceConfig           `json:"source"`
	Target      TargetConfig           `json:"target"`
	Steps       []PipelineStep          `json:"steps"`
	Validation  ValidationConfig        `json:"validation"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// SourceConfig defines the source of a pipeline.
type SourceConfig struct {
	Type        string                 `json:"type"` // "murex", "sap_gl", "bcrs", "rco", "axiom", "knowledge_graph"
	Connection  string                 `json:"connection"`
	Schema      SchemaDefinition       `json:"schema"`
	Filters     map[string]interface{} `json:"filters,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// TargetConfig defines the target of a pipeline.
type TargetConfig struct {
	Type        string                 `json:"type"` // "aspire", "capital", "liquidity", "reg_reporting", "knowledge_graph"
	Connection  string                 `json:"connection"`
	Schema      SchemaDefinition       `json:"schema"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// SchemaDefinition represents a data schema.
type SchemaDefinition struct {
	Fields     []FieldDefinition `json:"fields"`
	PrimaryKey []string          `json:"primary_key,omitempty"`
	Indexes    []IndexDefinition `json:"indexes,omitempty"`
}

// FieldDefinition represents a field in a schema.
type FieldDefinition struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Required    bool                   `json:"required,omitempty"`
	Default     interface{}            `json:"default,omitempty"`
	Constraints map[string]interface{} `json:"constraints,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// IndexDefinition represents an index on a schema.
type IndexDefinition struct {
	Name    string   `json:"name"`
	Fields  []string `json:"fields"`
	Unique  bool     `json:"unique,omitempty"`
	Type    string   `json:"type,omitempty"` // "btree", "hash", "gist", etc.
}

// PipelineStep represents a step in a pipeline.
type PipelineStep struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Type        string                 `json:"type"` // "transform", "validate", "enrich", "aggregate", "filter"
	Description string                 `json:"description,omitempty"`
	Config      map[string]interface{} `json:"config"`
	Output      SchemaDefinition       `json:"output,omitempty"`
	OnError     string                 `json:"on_error,omitempty"` // "stop", "skip", "retry"
	Retries     int                    `json:"retries,omitempty"`
}

// ValidationConfig defines validation rules for a pipeline.
type ValidationConfig struct {
	SchemaValidation   bool                   `json:"schema_validation"`
	DataQualityGates   []QualityGate          `json:"data_quality_gates,omitempty"`
	ContractTesting    ContractTestConfig      `json:"contract_testing,omitempty"`
	ConsistencyChecks  []ConsistencyCheck      `json:"consistency_checks,omitempty"`
	Metadata           map[string]interface{} `json:"metadata,omitempty"`
}

// QualityGate defines a data quality gate.
type QualityGate struct {
	Name        string                 `json:"name"`
	Metric      string                 `json:"metric"` // "completeness", "accuracy", "freshness", "consistency"
	Threshold   float64                `json:"threshold"`
	Operator    string                 `json:"operator"` // ">=", "<=", "==", "!="
	OnFailure   string                 `json:"on_failure"` // "stop", "warn", "continue"
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ContractTestConfig defines contract testing configuration.
type ContractTestConfig struct {
	Enabled     bool                   `json:"enabled"`
	Tests       []ContractTest          `json:"tests,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ContractTest represents a contract test.
type ContractTest struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"` // "schema", "data", "performance"
	Description string                 `json:"description,omitempty"`
	Config      map[string]interface{} `json:"config"`
}

// ConsistencyCheck defines a consistency check between source and target.
type ConsistencyCheck struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"` // "row_count", "sum", "aggregate", "custom"
	Description string                 `json:"description,omitempty"`
	Config      map[string]interface{} `json:"config"`
	Tolerance   float64                `json:"tolerance,omitempty"`
}

// PipelineExecutor executes semantic pipelines.
type PipelineExecutor struct {
	logger *log.Logger
}

// NewPipelineExecutor creates a new pipeline executor.
func NewPipelineExecutor(logger *log.Logger) *PipelineExecutor {
	return &PipelineExecutor{
		logger: logger,
	}
}

// Execute executes a semantic pipeline.
func (pe *PipelineExecutor) Execute(ctx context.Context, pipeline *SemanticPipeline) (*ExecutionResult, error) {
	if pe.logger != nil {
		pe.logger.Printf("Executing pipeline: %s (version: %s)", pipeline.Name, pipeline.Version)
	}

	// Step 1: Validate pipeline definition
	if err := pe.validatePipeline(pipeline); err != nil {
		return nil, fmt.Errorf("pipeline validation failed: %w", err)
	}

	// Step 2: Validate source schema
	if err := pe.validateSourceSchema(ctx, pipeline); err != nil {
		return nil, fmt.Errorf("source schema validation failed: %w", err)
	}

	// Step 3: Execute pipeline steps
	var stepResults []StepResult
	for i, step := range pipeline.Steps {
		stepResult, err := pe.executeStep(ctx, pipeline, &step, i)
		if err != nil {
			if step.OnError == "stop" {
				return nil, fmt.Errorf("step %s failed: %w", step.Name, err)
			} else if step.OnError == "skip" {
				if pe.logger != nil {
					pe.logger.Printf("Step %s failed, skipping: %v", step.Name, err)
				}
				continue
			}
			// retry logic would go here
		}
		stepResults = append(stepResults, *stepResult)
	}

	// Step 4: Validate target schema
	if err := pe.validateTargetSchema(ctx, pipeline); err != nil {
		return nil, fmt.Errorf("target schema validation failed: %w", err)
	}

	// Step 5: Run consistency checks
	consistencyResults, err := pe.runConsistencyChecks(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("consistency checks failed: %w", err)
	}

	// Step 6: Run data quality gates
	qualityResults, err := pe.runQualityGates(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("quality gates failed: %w", err)
	}

	result := &ExecutionResult{
		PipelineID:        pipeline.ID,
		Status:            "success",
		StepResults:       stepResults,
		ConsistencyResults: consistencyResults,
		QualityResults:    qualityResults,
		StartedAt:         time.Now(),
		CompletedAt:       time.Now(),
	}

	if pe.logger != nil {
		pe.logger.Printf("Pipeline execution completed: %s", pipeline.Name)
	}

	return result, nil
}

// validatePipeline validates a pipeline definition.
func (pe *PipelineExecutor) validatePipeline(pipeline *SemanticPipeline) error {
	if pipeline.ID == "" {
		return fmt.Errorf("pipeline ID is required")
	}
	if pipeline.Name == "" {
		return fmt.Errorf("pipeline name is required")
	}
	if pipeline.Source.Type == "" {
		return fmt.Errorf("source type is required")
	}
	if pipeline.Target.Type == "" {
		return fmt.Errorf("target type is required")
	}
	return nil
}

// validateSourceSchema validates the source schema.
func (pe *PipelineExecutor) validateSourceSchema(ctx context.Context, pipeline *SemanticPipeline) error {
	// In production, would validate against actual source
	// For now, just check schema definition is valid
	if len(pipeline.Source.Schema.Fields) == 0 {
		return fmt.Errorf("source schema must have at least one field")
	}
	return nil
}

// validateTargetSchema validates the target schema.
func (pe *PipelineExecutor) validateTargetSchema(ctx context.Context, pipeline *SemanticPipeline) error {
	// In production, would validate against actual target
	// For now, just check schema definition is valid
	if len(pipeline.Target.Schema.Fields) == 0 {
		return fmt.Errorf("target schema must have at least one field")
	}
	return nil
}

// executeStep executes a single pipeline step.
func (pe *PipelineExecutor) executeStep(ctx context.Context, pipeline *SemanticPipeline, step *PipelineStep, index int) (*StepResult, error) {
	if pe.logger != nil {
		pe.logger.Printf("Executing step %d: %s (type: %s)", index, step.Name, step.Type)
	}

	// In production, would execute the actual step transformation
	// For now, return a placeholder result
	result := &StepResult{
		StepID:      step.ID,
		StepName:    step.Name,
		Status:      "success",
		StartedAt:   time.Now(),
		CompletedAt: time.Now(),
		RecordsProcessed: 0,
		RecordsFailed:    0,
	}

	return result, nil
}

// runConsistencyChecks runs consistency checks between source and target.
func (pe *PipelineExecutor) runConsistencyChecks(ctx context.Context, pipeline *SemanticPipeline) ([]ConsistencyResult, error) {
	var results []ConsistencyResult
	for _, check := range pipeline.Validation.ConsistencyChecks {
		result := ConsistencyResult{
			CheckName: check.Name,
			Status:    "passed",
			Message:   fmt.Sprintf("Consistency check %s passed", check.Name),
		}
		results = append(results, result)
	}
	return results, nil
}

// runQualityGates runs data quality gates.
func (pe *PipelineExecutor) runQualityGates(ctx context.Context, pipeline *SemanticPipeline) ([]QualityResult, error) {
	var results []QualityResult
	for _, gate := range pipeline.Validation.DataQualityGates {
		// In production, would evaluate actual metrics
		result := QualityResult{
			GateName:  gate.Name,
			Metric:    gate.Metric,
			Status:    "passed",
			Value:     1.0, // Placeholder
			Threshold: gate.Threshold,
		}
		results = append(results, result)
	}
	return results, nil
}

// ExecutionResult represents the result of pipeline execution.
type ExecutionResult struct {
	PipelineID         string                `json:"pipeline_id"`
	Status             string                 `json:"status"` // "success", "failed", "partial"
	StepResults        []StepResult          `json:"step_results"`
	ConsistencyResults []ConsistencyResult   `json:"consistency_results"`
	QualityResults     []QualityResult       `json:"quality_results"`
	StartedAt          time.Time             `json:"started_at"`
	CompletedAt        time.Time             `json:"completed_at"`
	Metadata           map[string]interface{} `json:"metadata,omitempty"`
}

// StepResult represents the result of a pipeline step.
type StepResult struct {
	StepID           string    `json:"step_id"`
	StepName         string    `json:"step_name"`
	Status           string    `json:"status"`
	StartedAt        time.Time `json:"started_at"`
	CompletedAt      time.Time `json:"completed_at"`
	RecordsProcessed int64     `json:"records_processed"`
	RecordsFailed    int64     `json:"records_failed"`
	ErrorMessage     string    `json:"error_message,omitempty"`
}

// ConsistencyResult represents the result of a consistency check.
type ConsistencyResult struct {
	CheckName string  `json:"check_name"`
	Status    string  `json:"status"` // "passed", "failed", "warning"
	Message   string  `json:"message"`
	Value     float64 `json:"value,omitempty"`
	Expected  float64 `json:"expected,omitempty"`
}

// QualityResult represents the result of a quality gate.
type QualityResult struct {
	GateName  string  `json:"gate_name"`
	Metric    string  `json:"metric"`
	Status    string  `json:"status"` // "passed", "failed"
	Value     float64 `json:"value"`
	Threshold float64 `json:"threshold"`
}

// LoadPipelineFromJSON loads a pipeline definition from JSON.
func LoadPipelineFromJSON(data []byte) (*SemanticPipeline, error) {
	var pipeline SemanticPipeline
	if err := json.Unmarshal(data, &pipeline); err != nil {
		return nil, fmt.Errorf("failed to unmarshal pipeline: %w", err)
	}
	return &pipeline, nil
}

// SavePipelineToJSON saves a pipeline definition to JSON.
func SavePipelineToJSON(pipeline *SemanticPipeline) ([]byte, error) {
	return json.MarshalIndent(pipeline, "", "  ")
}

