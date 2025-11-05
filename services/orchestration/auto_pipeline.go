package orchestration

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sync"
	"time"
)

// AutoPipelineOrchestrator orchestrates end-to-end automation of extraction → training → deployment.
type AutoPipelineOrchestrator struct {
	logger            *log.Logger
	extractServiceURL string
	trainingServiceURL string
	gleanClient      *GleanClient
	modelRegistry    *ModelRegistry
	abTestManager    *ABTestManager
}

// NewAutoPipelineOrchestrator creates a new auto-pipeline orchestrator.
func NewAutoPipelineOrchestrator(logger *log.Logger) *AutoPipelineOrchestrator {
	return &AutoPipelineOrchestrator{
		logger:             logger,
		extractServiceURL:  os.Getenv("EXTRACT_SERVICE_URL"),
		trainingServiceURL: os.Getenv("TRAINING_SERVICE_URL"),
		gleanClient:       NewGleanClient(logger),
		modelRegistry:     NewModelRegistry(logger),
		abTestManager:     NewABTestManager(logger),
	}
}

// TriggerTrainingOnNewData automatically triggers training when new data arrives.
func (apo *AutoPipelineOrchestrator) TriggerTrainingOnNewData(
	ctx context.Context,
	projectID string,
	systemID string,
) error {
	apo.logger.Printf("Auto-triggering training for project=%s, system=%s", projectID, systemID)
	
	// Check for new data
	hasNewData, err := apo.checkForNewData(ctx, projectID, systemID)
	if err != nil {
		return fmt.Errorf("failed to check for new data: %w", err)
	}
	
	if !hasNewData {
		apo.logger.Printf("No new data detected, skipping training trigger")
		return nil
	}
	
	// Trigger training pipeline
	trainingResult, err := apo.runTrainingPipeline(ctx, projectID, systemID)
	if err != nil {
		return fmt.Errorf("training pipeline failed: %w", err)
	}
	
	// Deploy model if training successful
	if trainingResult.Success {
		deploymentResult, err := apo.deployModel(ctx, trainingResult.ModelPath)
		if err != nil {
			apo.logger.Printf("Model deployment failed: %v", err)
			return err
		}
		
		apo.logger.Printf("Model deployed successfully: %s", deploymentResult.Version)
	}
	
	return nil
}

// checkForNewData checks if new data has arrived.
func (apo *AutoPipelineOrchestrator) checkForNewData(
	ctx context.Context,
	projectID string,
	systemID string,
) (bool, error) {
	// Query Glean for recent data
	// This would query the knowledge graph or Glean Catalog for timestamps
	// For now, return true as a placeholder
	return true, nil
}

// runTrainingPipeline runs the training pipeline.
func (apo *AutoPipelineOrchestrator) runTrainingPipeline(
	ctx context.Context,
	projectID string,
	systemID string,
) (*TrainingResult, error) {
	apo.logger.Printf("Running training pipeline...")
	
	// Call training service
	cmd := exec.CommandContext(ctx, "python3", "./tools/scripts/train_relational_transformer.py",
		"--training-pipeline-enable",
		"--extract-project-id", projectID,
		"--extract-system-id", systemID,
	)
	
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("training pipeline failed: %w", err)
	}
	
	var result TrainingResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse training result: %w", err)
	}
	
	return &result, nil
}

// deployModel deploys a trained model.
func (apo *AutoPipelineOrchestrator) deployModel(
	ctx context.Context,
	modelPath string,
) (*DeploymentResult, error) {
	apo.logger.Printf("Deploying model from %s", modelPath)
	
	// Register model in registry
	version, err := apo.modelRegistry.RegisterModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to register model: %w", err)
	}
	
	// Start A/B test if enabled
	if os.Getenv("ENABLE_AB_TESTING") == "true" {
		err = apo.abTestManager.StartABTest(version)
		if err != nil {
			apo.logger.Printf("A/B test start failed: %v", err)
		}
	}
	
	return &DeploymentResult{
		Version:    version,
		ModelPath:  modelPath,
		DeployedAt: time.Now(),
	}, nil
}

// TrainingResult represents the result of a training run.
type TrainingResult struct {
	Success   bool   `json:"success"`
	ModelPath string `json:"model_path"`
	Metrics   map[string]any `json:"metrics"`
}

// DeploymentResult represents the result of a model deployment.
type DeploymentResult struct {
	Version    string    `json:"version"`
	ModelPath  string    `json:"model_path"`
	DeployedAt time.Time `json:"deployed_at"`
}

// ModelRegistry manages model versions.
type ModelRegistry struct {
	logger *log.Logger
	models map[string]ModelVersion
	mu     sync.RWMutex
}

// NewModelRegistry creates a new model registry.
func NewModelRegistry(logger *log.Logger) *ModelRegistry {
	return &ModelRegistry{
		logger: logger,
		models: make(map[string]ModelVersion),
	}
}

// ModelVersion represents a model version.
type ModelVersion struct {
	Version     string    `json:"version"`
	ModelPath   string    `json:"model_path"`
	Metrics     map[string]any `json:"metrics"`
	CreatedAt   time.Time `json:"created_at"`
	IsActive    bool      `json:"is_active"`
}

// RegisterModel registers a new model version.
func (mr *ModelRegistry) RegisterModel(modelPath string) (string, error) {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	version := fmt.Sprintf("v%d", len(mr.models)+1)
	
	modelVersion := ModelVersion{
		Version:   version,
		ModelPath: modelPath,
		CreatedAt: time.Now(),
		IsActive:  false, // New models start inactive for A/B testing
	}
	
	mr.models[version] = modelVersion
	mr.logger.Printf("Registered model version: %s", version)
	
	return version, nil
}

// ABTestManager manages A/B testing for models.
type ABTestManager struct {
	logger      *log.Logger
	activeTests map[string]*ABTest
	mu          sync.RWMutex
}

// NewABTestManager creates a new A/B test manager.
func NewABTestManager(logger *log.Logger) *ABTestManager {
	return &ABTestManager{
		logger:      logger,
		activeTests: make(map[string]*ABTest),
	}
}

// ABTest represents an A/B test.
type ABTest struct {
	TestID        string    `json:"test_id"`
	ControlVersion string   `json:"control_version"`
	TreatmentVersion string `json:"treatment_version"`
	StartTime     time.Time `json:"start_time"`
	TrafficSplit  float64   `json:"traffic_split"` // 0.0 to 1.0
	Metrics       map[string]any `json:"metrics"`
}

// StartABTest starts an A/B test for a new model version.
func (abtm *ABTestManager) StartABTest(newVersion string) error {
	abtm.mu.Lock()
	defer abtm.mu.Unlock()
	
	// Find current active version as control
	controlVersion := "v1" // Would find actual active version
	
	testID := fmt.Sprintf("ab_test_%s_%s", controlVersion, newVersion)
	
	test := &ABTest{
		TestID:          testID,
		ControlVersion:  controlVersion,
		TreatmentVersion: newVersion,
		StartTime:       time.Now(),
		TrafficSplit:    0.1, // Start with 10% traffic to new version
		Metrics:         make(map[string]any),
	}
	
	abtm.activeTests[testID] = test
	abtm.logger.Printf("Started A/B test: %s", testID)
	
	return nil
}

// GleanClient is a placeholder for Glean client.
type GleanClient struct {
	logger *log.Logger
}

// NewGleanClient creates a new Glean client.
func NewGleanClient(logger *log.Logger) *GleanClient {
	return &GleanClient{logger: logger}
}

// AutoRollback automatically rolls back on performance degradation.
func (apo *AutoPipelineOrchestrator) AutoRollback(
	ctx context.Context,
	version string,
	performanceThreshold float64,
) error {
	apo.logger.Printf("Checking performance for version %s", version)
	
	// Get performance metrics
	metrics, err := apo.modelRegistry.GetPerformanceMetrics(version)
	if err != nil {
		return fmt.Errorf("failed to get performance metrics: %w", err)
	}
	
	// Check if performance degraded
	performanceScore := metrics["accuracy"].(float64) // Simplified
	if performanceScore < performanceThreshold {
		apo.logger.Printf("Performance degradation detected: %.2f < %.2f", performanceScore, performanceThreshold)
		
		// Rollback to previous version
		err = apo.modelRegistry.RollbackToPreviousVersion(version)
		if err != nil {
			return fmt.Errorf("rollback failed: %w", err)
		}
		
		apo.logger.Printf("Rolled back to previous version")
	}
	
	return nil
}

// GetPerformanceMetrics gets performance metrics for a model version.
func (mr *ModelRegistry) GetPerformanceMetrics(version string) (map[string]any, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	
	modelVersion, exists := mr.models[version]
	if !exists {
		return nil, fmt.Errorf("model version not found: %s", version)
	}
	
	return modelVersion.Metrics, nil
}

// RollbackToPreviousVersion rolls back to the previous model version.
func (mr *ModelRegistry) RollbackToPreviousVersion(currentVersion string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	// Find previous active version
	for version, modelVersion := range mr.models {
		if version != currentVersion && modelVersion.IsActive {
			// Deactivate current version
			if current, exists := mr.models[currentVersion]; exists {
				current.IsActive = false
				mr.models[currentVersion] = current
			}
			
			// Activate previous version
			modelVersion.IsActive = true
			mr.models[version] = modelVersion
			
			mr.logger.Printf("Rolled back from %s to %s", currentVersion, version)
			return nil
		}
	}
	
	return fmt.Errorf("no previous version found for rollback")
}

