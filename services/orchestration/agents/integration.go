package agents

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"
)

// AgentSystem integrates all agents into a coordinated system.
type AgentSystem struct {
	coordinator      interface{} // *orchestration.AgentCoordinator - using interface to avoid circular dependency
	factory          *AgentFactory
	ingestionAgents  map[string]*DataIngestionAgent
	mappingAgent     *MappingRuleAgent
	anomalyAgent    *AnomalyDetectionAgent
	testAgent        *TestGenerationAgent
	logger           *log.Logger
}

// NewAgentSystem creates a new agent system with all agents.
func NewAgentSystem(
	coordinator interface{}, // *orchestration.AgentCoordinator
	graphClient GraphClient,
	db *sql.DB,
	logger *log.Logger,
) *AgentSystem {
	// Create rule store and alert manager
	ruleStore := NewPostgresMappingRuleStore(db, logger)
	alertManager := NewDefaultAlertManager(logger)

	// Create factory
	factory := NewAgentFactory(graphClient, ruleStore, alertManager, logger)

	// Create agents
	system := &AgentSystem{
		coordinator:     coordinator,
		factory:         factory,
		ingestionAgents: make(map[string]*DataIngestionAgent),
		mappingAgent:    factory.CreateMappingRuleAgent(),
		anomalyAgent:    factory.CreateAnomalyDetectionAgent(),
		testAgent:       factory.CreateTestGenerationAgent(),
		logger:          logger,
	}

	// Register agents with coordinator (would call coordinator methods in production)
	// coordinator.RegisterAgent("mapping-rule-agent", "mapping_rule")
	// coordinator.RegisterAgent("anomaly-detection-agent", "anomaly_detection")
	// coordinator.RegisterAgent("test-generation-agent", "test_generation")

	return system
}

// RegisterIngestionAgent registers a data ingestion agent for a source type.
func (as *AgentSystem) RegisterIngestionAgent(sourceType string, config map[string]interface{}) error {
	agent, err := as.factory.CreateDataIngestionAgent(sourceType, config)
	if err != nil {
		return err
	}

	as.ingestionAgents[sourceType] = agent
	// as.coordinator.RegisterAgent(agent.ID, "data_ingestion") // Would register in production

	if as.logger != nil {
		as.logger.Printf("Registered ingestion agent for %s", sourceType)
	}

	return nil
}

// RunIngestion runs data ingestion for a source type.
func (as *AgentSystem) RunIngestion(ctx context.Context, sourceType string, config map[string]interface{}) error {
	agent, exists := as.ingestionAgents[sourceType]
	if !exists {
		return fmt.Errorf("ingestion agent not registered for %s", sourceType)
	}

	return agent.Ingest(ctx, config)
}

// RunAnomalyDetection runs anomaly detection on data.
func (as *AgentSystem) RunAnomalyDetection(ctx context.Context, data []DataPoint) ([]Anomaly, error) {
	return as.anomalyAgent.DetectAnomalies(ctx, data)
}

// GenerateTests generates and optionally runs test scenarios.
func (as *AgentSystem) GenerateTests(ctx context.Context, schema interface{}, options TestGenOptions) ([]TestResult, error) {
	return as.testAgent.GenerateAndRunTests(ctx, schema, options)
}

// UpdateMappingRules updates mapping rules from patterns.
func (as *AgentSystem) UpdateMappingRules(ctx context.Context, patterns []MappingPattern) error {
	return as.mappingAgent.LearnAndUpdate(ctx, patterns)
}

// GetSystemStatus returns the status of all agents.
func (as *AgentSystem) GetSystemStatus() map[string]interface{} {
	status := map[string]interface{}{
		"ingestion_agents": len(as.ingestionAgents),
		"mapping_agent":    as.mappingAgent.GetStats(),
		"anomaly_agent":    as.anomalyAgent.GetStats(),
		"test_agent":       as.testAgent.GetStats(),
	}

	return status
}

