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
// Phase 9.3: Enhanced with domain-aware orchestration for domain-specific pipelines.
type AutoPipelineOrchestrator struct {
	logger            *log.Logger
	extractServiceURL string
	trainingServiceURL string
	gleanClient      *GleanClient
	modelRegistry    *ModelRegistry
	abTestManager    *ABTestManager
	domainDetector   *DomainDetector // Phase 9.3: Domain detector for domain-aware orchestration
}

// DomainDetector is a placeholder - would import from extract service
type DomainDetector struct {
	localaiURL string
	logger     *log.Logger
}

// NewDomainDetector creates a domain detector (placeholder).
func NewDomainDetector(localaiURL string, logger *log.Logger) *DomainDetector {
	return &DomainDetector{
		localaiURL: localaiURL,
		logger:     logger,
	}
}

// NewAutoPipelineOrchestrator creates a new auto-pipeline orchestrator.
func NewAutoPipelineOrchestrator(logger *log.Logger) *AutoPipelineOrchestrator {
	localaiURL := os.Getenv("LOCALAI_URL")
	var domainDetector *DomainDetector
	if localaiURL != "" {
		domainDetector = NewDomainDetector(localaiURL, logger)
	}
	
	return &AutoPipelineOrchestrator{
		logger:             logger,
		extractServiceURL:  os.Getenv("EXTRACT_SERVICE_URL"),
		trainingServiceURL: os.Getenv("TRAINING_SERVICE_URL"),
		gleanClient:       NewGleanClient(logger),
		modelRegistry:     NewModelRegistry(logger),
		abTestManager:     NewABTestManager(logger),
		domainDetector:    domainDetector, // Phase 9.3: Domain detector
	}
}

// TriggerTrainingOnNewData automatically triggers training when new data arrives.
// Phase 9.3: Enhanced with domain-aware training orchestration.
func (apo *AutoPipelineOrchestrator) TriggerTrainingOnNewData(
	ctx context.Context,
	projectID string,
	systemID string,
	domainID string, // Phase 9.3: Optional domain ID for domain-specific training
) error {
	apo.logger.Printf(
		"Auto-triggering training for project=%s, system=%s, domain=%s",
		projectID, systemID, domainID,
	)
	
	// Check for new data
	hasNewData, err := apo.checkForNewData(ctx, projectID, systemID)
	if err != nil {
		return fmt.Errorf("failed to check for new data: %w", err)
	}
	
	if !hasNewData {
		apo.logger.Printf("No new data detected, skipping training trigger")
		return nil
	}
	
	// Phase 9.3: Auto-detect domain if not provided
	if domainID == "" && apo.domainDetector != nil {
		// Try to detect domain from project/system metadata
		// This would query the knowledge graph or use domain detector
		domainID = "" // Placeholder - would implement domain detection
	}
	
	// Trigger training pipeline with domain
	trainingResult, err := apo.runTrainingPipeline(ctx, projectID, systemID, domainID)
	if err != nil {
		return fmt.Errorf("training pipeline failed: %w", err)
	}
	
	// Deploy model if training successful (with domain context)
	if trainingResult.Success {
		deploymentResult, err := apo.deployModel(ctx, trainingResult.ModelPath, domainID)
		if err != nil {
			apo.logger.Printf("Model deployment failed: %v", err)
			return err
		}
		
		apo.logger.Printf(
			"Model deployed successfully: %s (domain: %s)",
			deploymentResult.Version, domainID,
		)
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
// Phase 9.3: Enhanced with domain-aware training configuration.
func (apo *AutoPipelineOrchestrator) runTrainingPipeline(
	ctx context.Context,
	projectID string,
	systemID string,
	domainID string, // Phase 9.3: Domain ID for domain-specific training
) (*TrainingResult, error) {
	apo.logger.Printf("Running training pipeline (domain: %s)...", domainID)
	
	// Build command arguments
	args := []string{
		"./tools/scripts/train_relational_transformer.py",
		"--training-pipeline-enable",
		"--extract-project-id", projectID,
		"--extract-system-id", systemID,
	}
	
	// Phase 9.3: Add domain-specific arguments if domainID provided
	if domainID != "" {
		args = append(args, "--domain-id", domainID)
		args = append(args, "--enable-domain-filtering", "true")
	}
	
	// Call training service
	cmd := exec.CommandContext(ctx, "python3", args...)
	
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("training pipeline failed: %w", err)
	}
	
	var result TrainingResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse training result: %w", err)
	}
	
	// Phase 9.3: Add domain context to result
	if domainID != "" {
		result.DomainID = domainID
	}
	
	return &result, nil
}

// deployModel deploys a trained model.
// Phase 9.3: Enhanced with domain-aware deployment.
func (apo *AutoPipelineOrchestrator) deployModel(
	ctx context.Context,
	modelPath string,
	domainID string, // Phase 9.3: Domain ID for domain-specific deployment
) (*DeploymentResult, error) {
	apo.logger.Printf("Deploying model from %s (domain: %s)", modelPath, domainID)
	
	// Register model in registry (with domain context)
	version, err := apo.modelRegistry.RegisterModel(modelPath, domainID)
	if err != nil {
		return nil, fmt.Errorf("failed to register model: %w", err)
	}
	
	// Phase 9.3: Start domain-specific A/B test if enabled
	if os.Getenv("ENABLE_AB_TESTING") == "true" {
		err = apo.abTestManager.StartABTest(version, domainID)
		if err != nil {
			apo.logger.Printf("A/B test start failed: %v", err)
		}
	}
	
	return &DeploymentResult{
		Version:    version,
		ModelPath:  modelPath,
		DomainID:   domainID, // Phase 9.3: Include domain in result
		DeployedAt: time.Now(),
	}, nil
}

// TrainingResult represents the result of a training run.
type TrainingResult struct {
	Success   bool   `json:"success"`
	ModelPath string `json:"model_path"`
	Metrics   map[string]any `json:"metrics"`
	DomainID  string `json:"domain_id,omitempty"` // Phase 9.3: Domain context
}

// DeploymentResult represents the result of a model deployment.
type DeploymentResult struct {
	Version    string    `json:"version"`
	ModelPath  string    `json:"model_path"`
	DomainID   string    `json:"domain_id,omitempty"` // Phase 9.3: Domain context
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
	DomainID    string    `json:"domain_id,omitempty"` // Phase 9.3: Domain context
	Metrics     map[string]any `json:"metrics"`
	CreatedAt   time.Time `json:"created_at"`
	IsActive    bool      `json:"is_active"`
}

// RegisterModel registers a new model version.
// Phase 9.3: Enhanced with domain context.
func (mr *ModelRegistry) RegisterModel(modelPath string, domainID string) (string, error) {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	version := fmt.Sprintf("v%d", len(mr.models)+1)
	
	modelVersion := ModelVersion{
		Version:   version,
		ModelPath: modelPath,
		DomainID:  domainID, // Phase 9.3: Domain context
		CreatedAt: time.Now(),
		IsActive:  false, // New models start inactive for A/B testing
	}
	
	mr.models[version] = modelVersion
	mr.logger.Printf("Registered model version: %s (domain: %s)", version, domainID)
	
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
	DomainID      string    `json:"domain_id,omitempty"` // Phase 9.3: Domain context
	ControlVersion string   `json:"control_version"`
	TreatmentVersion string `json:"treatment_version"`
	StartTime     time.Time `json:"start_time"`
	TrafficSplit  float64   `json:"traffic_split"` // 0.0 to 1.0
	Metrics       map[string]any `json:"metrics"`
}

// StartABTest starts an A/B test for a new model version.
// Phase 9.3: Enhanced with domain-specific A/B testing.
func (abtm *ABTestManager) StartABTest(newVersion string, domainID string) error {
	abtm.mu.Lock()
	defer abtm.mu.Unlock()
	
	// Find current active version as control (for this domain)
	controlVersion := "v1" // Would find actual active version for domain
	
	testID := fmt.Sprintf("ab_test_%s_%s_%s", domainID, controlVersion, newVersion)
	
	test := &ABTest{
		TestID:          testID,
		DomainID:       domainID, // Phase 9.3: Domain context
		ControlVersion:  controlVersion,
		TreatmentVersion: newVersion,
		StartTime:       time.Now(),
		TrafficSplit:    0.1, // Start with 10% traffic to new version
		Metrics:         make(map[string]any),
	}
	
	abtm.activeTests[testID] = test
	abtm.logger.Printf("Started A/B test: %s (domain: %s)", testID, domainID)
	
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

