package agents

import (
	"context"
	"database/sql"
	"log"
	"os"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

func setupTestDB(t *testing.T) *sql.DB {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open test database: %v", err)
	}

	// Create minimal schema for testing
	schema := `
		CREATE TABLE IF NOT EXISTS mapping_rules (
			source_type TEXT,
			version TEXT,
			rules_json TEXT,
			confidence FLOAT,
			created_at TIMESTAMP,
			PRIMARY KEY (source_type, version)
		);
	`
	if _, err := db.Exec(schema); err != nil {
		t.Fatalf("Failed to create schema: %v", err)
	}

	return db
}

func TestDataIngestionAgent_Ingest(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	
	// Create mock graph client
	graphClient := &MockGraphClient{}
	
	// Create mapper
	mapper := NewDefaultSchemaMapper(logger)
	
	// Create connector
	connector := NewMurexConnector(map[string]interface{}{}, logger)
	
	// Create agent
	agent := NewDataIngestionAgent(
		"test-agent",
		"murex",
		connector,
		mapper,
		graphClient,
		logger,
	)

	ctx := context.Background()
	config := map[string]interface{}{
		"connection_string": "test",
	}

	err := agent.Ingest(ctx, config)
	if err != nil {
		t.Errorf("Ingest() error = %v", err)
	}

	stats := agent.GetStats()
	if stats.TotalRuns == 0 {
		t.Error("Expected at least one run")
	}
}

func TestMappingRuleAgent_LearnAndUpdate(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	ruleStore := NewPostgresMappingRuleStore(db, logger)
	learner := NewDefaultRuleLearner(logger)
	
	agent := NewMappingRuleAgent("test-agent", ruleStore, learner, logger)

	ctx := context.Background()
	patterns := []MappingPattern{
		{
			SourceTable:      "trades",
			SourceColumns:     []string{"trade_id", "amount"},
			TargetLabel:       "Trade",
			TargetProperties:  []string{"id", "value"},
			SuccessCount:      10,
			FailureCount:     0,
		},
	}

	err := agent.LearnAndUpdate(ctx, patterns)
	if err != nil {
		t.Errorf("LearnAndUpdate() error = %v", err)
	}

	stats := agent.GetStats()
	if stats.TotalUpdates == 0 {
		t.Error("Expected at least one update")
	}
}

func TestAnomalyDetectionAgent_DetectAnomalies(t *testing.T) {
	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	
	detectors := []AnomalyDetector{
		NewStatisticalAnomalyDetector(3.0, logger),
		NewPatternAnomalyDetector(logger),
	}
	
	alertManager := NewDefaultAlertManager(logger)
	graphClient := &MockGraphClient{}
	
	agent := NewAnomalyDetectionAgent(
		"test-agent",
		detectors,
		alertManager,
		graphClient,
		logger,
	)

	ctx := context.Background()
	dataPoints := []DataPoint{
		{Timestamp: time.Now(), Value: 100.0, Dimensions: map[string]interface{}{"field": "value"}},
		{Timestamp: time.Now().Add(1 * time.Second), Value: 150.0},
		{Timestamp: time.Now().Add(2 * time.Second), Value: 200.0},
	}

	anomalies, err := agent.DetectAnomalies(ctx, dataPoints)
	if err != nil {
		t.Errorf("DetectAnomalies() error = %v", err)
	}

	_ = anomalies // Would assert on anomalies
}

func TestTestGenerationAgent_GenerateAndRunTests(t *testing.T) {
	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	
	generator := NewDefaultTestScenarioGenerator(logger)
	orchestrator := NewDefaultTestOrchestrator(logger)
	
	agent := NewTestGenerationAgent("test-agent", generator, orchestrator, logger)

	ctx := context.Background()
	schema := map[string]interface{}{
		"fields": []map[string]interface{}{
			{"name": "id", "type": "string"},
			{"name": "value", "type": "number"},
		},
	}

	options := TestGenOptions{
		GenerateFromSchema: true,
		GenerateEdgeCases:  true,
		RunTests:           true,
	}

	results, err := agent.GenerateAndRunTests(ctx, schema, options)
	if err != nil {
		t.Errorf("GenerateAndRunTests() error = %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected test results")
	}
}

// MockGraphClient is a mock implementation of GraphClient.
type MockGraphClient struct{}

func (m *MockGraphClient) UpsertNodes(ctx context.Context, nodes []GraphNode) error {
	return nil
}

func (m *MockGraphClient) UpsertEdges(ctx context.Context, edges []GraphEdge) error {
	return nil
}

func (m *MockGraphClient) Query(ctx context.Context, cypher string, params map[string]interface{}) ([]map[string]interface{}, error) {
	return []map[string]interface{}{}, nil
}

