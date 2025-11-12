package autonomous

import (
	"context"
	"database/sql"
	"log"

	"github.com/plturrell/aModels/services/catalog/research"
	"github.com/pressly/goose/v3"
)

// IntegratedAutonomousSystem integrates IntelligenceLayer with Goose, Deep Research,
// DeepAgents, and Unified Workflow for complete autonomous operations.
type IntegratedAutonomousSystem struct {
	intelligenceLayer *IntelligenceLayer
	db                *sql.DB
	logger            *log.Logger
}

// NewIntegratedAutonomousSystem creates a new integrated autonomous system.
func NewIntegratedAutonomousSystem(
	deepResearchClient *research.DeepResearchClient,
	deepAgentsURL string,
	unifiedWorkflowURL string,
	db *sql.DB,
	logger *log.Logger,
) *IntegratedAutonomousSystem {
	intelligenceLayer := NewIntelligenceLayer(
		deepResearchClient,
		deepAgentsURL,
		unifiedWorkflowURL,
		db != nil, // Goose enabled if DB is available
		logger,
	)

	return &IntegratedAutonomousSystem{
		intelligenceLayer: intelligenceLayer,
		db:                db,
		logger:            logger,
	}
}

// ExecuteWithGooseMigration executes an autonomous task and records it via Goose migration.
func (ias *IntegratedAutonomousSystem) ExecuteWithGooseMigration(ctx context.Context, task *AutonomousTask) (*TaskResult, error) {
	// Execute autonomous task
	result, err := ias.intelligenceLayer.ExecuteAutonomousTask(ctx, task)
	if err != nil {
		return nil, err
	}

	// Record execution in database via Goose if available
	if ias.db != nil {
		if err := ias.recordExecutionInDB(ctx, task, result); err != nil {
			ias.logger.Printf("Failed to record execution in DB (non-fatal): %v", err)
		}
	}

	return result, nil
}

// recordExecutionInDB records task execution in the database.
func (ias *IntegratedAutonomousSystem) recordExecutionInDB(ctx context.Context, task *AutonomousTask, result *TaskResult) error {
	// This would create a migration file or record in a table
	// For now, we'll use a simple approach - in production, you'd use Goose migrations
	
	// Example: Insert into autonomous_task_executions table
	query := `
		INSERT INTO autonomous_task_executions (
			task_id, task_type, status, started_at, completed_at, 
			success, lessons_learned_count, optimizations_applied_count
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`
	
	// Note: This assumes the table exists via Goose migration
	// In practice, you'd create a migration file like:
	// 007_create_autonomous_task_executions.sql
	
	_, err := ias.db.ExecContext(ctx, query,
		task.ID,
		task.Type,
		"completed",
		taskContextValue(ctx, "started_at"),
		taskContextValue(ctx, "completed_at"),
		result.Success,
		len(result.Learned),
		len(result.Optimized),
	)
	
	return err
}

// taskContextValue extracts a value from context (simplified).
func taskContextValue(ctx context.Context, key string) interface{} {
	// In production, you'd use context values properly
	return nil
}

// RunGooseMigration runs a Goose migration for autonomous system tables.
func (ias *IntegratedAutonomousSystem) RunGooseMigration(ctx context.Context, migrationDir string) error {
	if ias.db == nil {
		return nil // No DB, skip migrations
	}

	// Run Goose migrations
	return goose.RunContext(ctx, "up", ias.db, migrationDir)
}

// GetPerformanceMetrics returns current performance metrics.
func (ias *IntegratedAutonomousSystem) GetPerformanceMetrics() *PerformanceMetrics {
	return ias.intelligenceLayer.performanceMetrics
}

// GetAgentRegistry returns the agent registry.
func (ias *IntegratedAutonomousSystem) GetAgentRegistry() *AgentRegistry {
	return ias.intelligenceLayer.agentRegistry
}

// GetKnowledgeBase returns the knowledge base.
func (ias *IntegratedAutonomousSystem) GetKnowledgeBase() *KnowledgeBase {
	return ias.intelligenceLayer.knowledgeBase
}

