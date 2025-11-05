package agents

import (
	"database/sql"
	"fmt"
	"log"
)

// AgentFactory creates and configures agents.
type AgentFactory struct {
	graphClient   GraphClient
	ruleStore     MappingRuleStore
	alertManager  AlertManager
	logger        *log.Logger
}

// NewAgentFactory creates a new agent factory.
func NewAgentFactory(
	graphClient GraphClient,
	ruleStore MappingRuleStore,
	alertManager AlertManager,
	logger *log.Logger,
) *AgentFactory {
	return &AgentFactory{
		graphClient:  graphClient,
		ruleStore:    ruleStore,
		alertManager: alertManager,
		logger:       logger,
	}
}

// CreateDataIngestionAgent creates a data ingestion agent for a source type.
func (af *AgentFactory) CreateDataIngestionAgent(sourceType string, config map[string]interface{}) (*DataIngestionAgent, error) {
	var connector SourceConnector

	// Create source-specific connector
	switch sourceType {
	case "murex":
		connector = NewMurexConnector(config, af.logger)
	case "sap_gl":
		connector = NewSAPGLConnector(config, af.logger)
	case "bcrs":
		connector = NewBCRSConnector(config, af.logger)
	case "rco":
		connector = NewRCOConnector(config, af.logger)
	case "axiom":
		connector = NewAxiomConnector(config, af.logger)
	default:
		return nil, fmt.Errorf("unknown source type: %s", sourceType)
	}

	mapper := NewDefaultSchemaMapper(af.logger)

	agent := NewDataIngestionAgent(
		fmt.Sprintf("ingestion-%s", sourceType),
		sourceType,
		connector,
		mapper,
		af.graphClient,
		af.logger,
	)

	return agent, nil
}

// CreateMappingRuleAgent creates a mapping rule agent.
func (af *AgentFactory) CreateMappingRuleAgent() *MappingRuleAgent {
	learner := NewDefaultRuleLearner(af.logger)

	return NewMappingRuleAgent(
		"mapping-rule-agent",
		af.ruleStore,
		learner,
		af.logger,
	)
}

// CreateAnomalyDetectionAgent creates an anomaly detection agent.
func (af *AgentFactory) CreateAnomalyDetectionAgent() *AnomalyDetectionAgent {
	detectors := []AnomalyDetector{
		NewStatisticalAnomalyDetector(3.0, af.logger), // 3-sigma threshold
		NewPatternAnomalyDetector(af.logger),
	}

	return NewAnomalyDetectionAgent(
		"anomaly-detection-agent",
		detectors,
		af.alertManager,
		af.graphClient,
		af.logger,
	)
}

// CreateTestGenerationAgent creates a test generation agent.
func (af *AgentFactory) CreateTestGenerationAgent() *TestGenerationAgent {
	generator := NewDefaultTestScenarioGenerator(af.logger)
	orchestrator := NewDefaultTestOrchestrator(af.logger)

	return NewTestGenerationAgent(
		"test-generation-agent",
		generator,
		orchestrator,
		af.logger,
	)
}

