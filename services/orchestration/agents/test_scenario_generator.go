package agents

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"
)

// TestScenarioGenerator generates test scenarios based on data patterns and system configurations.
type TestScenarioGenerator struct {
	ID            string
	GraphClient   GraphClient
	RuleStore     MappingRuleStore
	logger        *log.Logger
	stats         GeneratorStats
}

// GeneratorStats tracks test scenario generation statistics.
type GeneratorStats struct {
	TotalScenariosGenerated int
	ScenariosByType         map[string]int
	LastGenerated           time.Time
}

// TestScenario represents a generated test scenario.
type TestScenario struct {
	ID              string
	Name            string
	Type            string // "data_quality", "regulatory", "integration", "performance", "edge_case"
	Description     string
	Steps           []ScenarioStep
	ExpectedResults map[string]interface{}
	DataPatterns   []DataPattern
	Priority        string // "low", "medium", "high", "critical"
	Tags            []string
	CreatedAt       time.Time
}

// ScenarioStep represents a step in a test scenario.
type ScenarioStep struct {
	Order       int
	Action      string
	Description string
	Parameters  map[string]interface{}
	Assertions  []Assertion
}

// Assertion represents an assertion in a test scenario.
type Assertion struct {
	Type        string // "equals", "contains", "greater_than", "exists", "not_null"
	Field       string
	ExpectedValue interface{}
	Description string
}

// DataPattern represents a data pattern used to generate scenarios.
type DataPattern struct {
	Type        string // "distribution", "correlation", "anomaly", "trend"
	Description string
	Parameters  map[string]interface{}
	Confidence  float64
}

// NewTestScenarioGenerator creates a new test scenario generator.
func NewTestScenarioGenerator(
	id string,
	graphClient GraphClient,
	ruleStore MappingRuleStore,
	logger *log.Logger,
) *TestScenarioGenerator {
	return &TestScenarioGenerator{
		ID:          id,
		GraphClient: graphClient,
		RuleStore:   ruleStore,
		logger:      logger,
		stats: GeneratorStats{
			ScenariosByType: make(map[string]int),
		},
	}
}

// GenerateScenarios generates test scenarios based on data patterns and system state.
func (tsg *TestScenarioGenerator) GenerateScenarios(ctx context.Context, config ScenarioGenerationConfig) ([]TestScenario, error) {
	tsg.stats.TotalScenariosGenerated += config.Count
	tsg.stats.LastGenerated = time.Now()

	if tsg.logger != nil {
		tsg.logger.Printf("Generating %d test scenarios (types: %v)", config.Count, config.Types)
	}

	var scenarios []TestScenario

	// Query knowledge graph for data patterns
	patterns, err := tsg.discoverDataPatterns(ctx)
	if err != nil {
		if tsg.logger != nil {
			tsg.logger.Printf("Warning: Failed to discover data patterns: %v", err)
		}
		// Continue with default patterns
		patterns = tsg.getDefaultPatterns()
	}

	// Generate scenarios based on requested types
	typesToGenerate := config.Types
	if len(typesToGenerate) == 0 {
		typesToGenerate = []string{"data_quality", "regulatory", "integration"}
	}

	for i := 0; i < config.Count; i++ {
		scenarioType := typesToGenerate[rand.Intn(len(typesToGenerate))]
		scenario, err := tsg.generateScenarioForType(ctx, scenarioType, patterns, i)
		if err != nil {
			if tsg.logger != nil {
				tsg.logger.Printf("Warning: Failed to generate scenario %d: %v", i, err)
			}
			continue
		}
		scenarios = append(scenarios, *scenario)
		tsg.stats.ScenariosByType[scenarioType]++
	}

	if tsg.logger != nil {
		tsg.logger.Printf("Generated %d test scenarios", len(scenarios))
	}

	return scenarios, nil
}

// ScenarioGenerationConfig configures test scenario generation.
type ScenarioGenerationConfig struct {
	Count int
	Types []string // "data_quality", "regulatory", "integration", "performance", "edge_case"
	Focus []string // Specific areas to focus on
}

// discoverDataPatterns queries the knowledge graph for data patterns.
func (tsg *TestScenarioGenerator) discoverDataPatterns(ctx context.Context) ([]DataPattern, error) {
	if tsg.GraphClient == nil {
		return tsg.getDefaultPatterns(), nil
	}

	// Query for common data patterns
	cypher := `
		MATCH (n)
		WHERE n.type IN ['Trade', 'JournalEntry', 'RegulatoryCalculation']
		WITH n.type as type, count(n) as count
		RETURN type, count
		ORDER BY count DESC
		LIMIT 10
	`

	results, err := tsg.GraphClient.Query(ctx, cypher, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query knowledge graph: %w", err)
	}

	var patterns []DataPattern
	for _, result := range results {
		nodeType, _ := result["type"].(string)
		count, _ := result["count"].(float64)
		
		patterns = append(patterns, DataPattern{
			Type:        "distribution",
			Description: fmt.Sprintf("High concentration of %s nodes (%d)", nodeType, int(count)),
			Parameters: map[string]interface{}{
				"node_type": nodeType,
				"count":     count,
			},
			Confidence: 0.8,
		})
	}

	if len(patterns) == 0 {
		return tsg.getDefaultPatterns(), nil
	}

	return patterns, nil
}

// getDefaultPatterns returns default data patterns when graph query fails.
func (tsg *TestScenarioGenerator) getDefaultPatterns() []DataPattern {
	return []DataPattern{
		{
			Type:        "distribution",
			Description: "Normal distribution of trade values",
			Parameters: map[string]interface{}{
				"mean":   1000000,
				"stddev": 500000,
			},
			Confidence: 0.7,
		},
		{
			Type:        "correlation",
			Description: "Trade volume correlates with market volatility",
			Parameters: map[string]interface{}{
				"correlation": 0.65,
			},
			Confidence: 0.6,
		},
	}
}

// generateScenarioForType generates a test scenario for a specific type.
func (tsg *TestScenarioGenerator) generateScenarioForType(
	ctx context.Context,
	scenarioType string,
	patterns []DataPattern,
	index int,
) (*TestScenario, error) {
	scenarioID := fmt.Sprintf("scenario-%s-%d-%d", scenarioType, time.Now().Unix(), index)

	switch scenarioType {
	case "data_quality":
		return tsg.generateDataQualityScenario(scenarioID, patterns)
	case "regulatory":
		return tsg.generateRegulatoryScenario(scenarioID, patterns)
	case "integration":
		return tsg.generateIntegrationScenario(scenarioID, patterns)
	case "performance":
		return tsg.generatePerformanceScenario(scenarioID, patterns)
	case "edge_case":
		return tsg.generateEdgeCaseScenario(scenarioID, patterns)
	default:
		return tsg.generateDataQualityScenario(scenarioID, patterns)
	}
}

// generateDataQualityScenario generates a data quality test scenario.
func (tsg *TestScenarioGenerator) generateDataQualityScenario(id string, patterns []DataPattern) (*TestScenario, error) {
	scenario := &TestScenario{
		ID:          id,
		Name:        "Data Quality Validation Test",
		Type:        "data_quality",
		Description: "Validates data quality metrics for trades and journal entries",
		Steps: []ScenarioStep{
			{
				Order:       1,
				Action:      "extract_data",
				Description: "Extract trade data from Murex",
				Parameters: map[string]interface{}{
					"source": "Murex",
					"table":  "trades",
					"limit":  1000,
				},
				Assertions: []Assertion{
					{Type: "not_null", Field: "trade_id", Description: "Trade ID must not be null"},
					{Type: "exists", Field: "trade_date", Description: "Trade date must exist"},
				},
			},
			{
				Order:       2,
				Action:      "validate_schema",
				Description: "Validate data against schema",
				Parameters: map[string]interface{}{
					"schema": "trade_schema_v1",
				},
				Assertions: []Assertion{
					{Type: "equals", Field: "schema_version", ExpectedValue: "v1", Description: "Schema version must match"},
				},
			},
			{
				Order:       3,
				Action:      "check_quality_metrics",
				Description: "Check data quality metrics",
				Parameters: map[string]interface{}{
					"metrics": []string{"completeness", "accuracy", "freshness"},
				},
				Assertions: []Assertion{
					{Type: "greater_than", Field: "completeness", ExpectedValue: 0.95, Description: "Completeness must be > 95%"},
				},
			},
		},
		ExpectedResults: map[string]interface{}{
			"quality_score": 0.95,
			"status":        "passed",
		},
		DataPatterns: patterns,
		Priority:     "high",
		Tags:         []string{"data-quality", "validation", "murex"},
		CreatedAt:    time.Now(),
	}

	return scenario, nil
}

// generateRegulatoryScenario generates a regulatory test scenario.
func (tsg *TestScenarioGenerator) generateRegulatoryScenario(id string, patterns []DataPattern) (*TestScenario, error) {
	scenario := &TestScenario{
		ID:          id,
		Name:        "Regulatory Calculation Validation Test",
		Type:        "regulatory",
		Description: "Validates regulatory calculations for MAS 610 compliance",
		Steps: []ScenarioStep{
			{
				Order:       1,
				Action:      "calculate_regulatory_metric",
				Description: "Calculate regulatory capital requirement",
				Parameters: map[string]interface{}{
					"framework": "MAS 610",
					"metric":    "capital_adequacy_ratio",
				},
				Assertions: []Assertion{
					{Type: "greater_than", Field: "result", ExpectedValue: 0.0, Description: "Result must be positive"},
				},
			},
			{
				Order:       2,
				Action:      "validate_against_rules",
				Description: "Validate calculation against regulatory rules",
				Parameters: map[string]interface{}{
					"rule_set": "MAS_610_RULES",
				},
				Assertions: []Assertion{
					{Type: "contains", Field: "validation_status", ExpectedValue: "passed", Description: "Validation must pass"},
				},
			},
		},
		ExpectedResults: map[string]interface{}{
			"calculation_status": "valid",
			"compliance":         true,
		},
		DataPatterns: patterns,
		Priority:     "critical",
		Tags:         []string{"regulatory", "MAS610", "compliance"},
		CreatedAt:    time.Now(),
	}

	return scenario, nil
}

// generateIntegrationScenario generates an integration test scenario.
func (tsg *TestScenarioGenerator) generateIntegrationScenario(id string, patterns []DataPattern) (*TestScenario, error) {
	scenario := &TestScenario{
		ID:          id,
		Name:        "System Integration Test",
		Type:        "integration",
		Description: "Tests integration between Murex and SAP GL",
		Steps: []ScenarioStep{
			{
				Order:       1,
				Action:      "ingest_from_murex",
				Description: "Ingest trade data from Murex",
				Parameters: map[string]interface{}{
					"source": "Murex",
				},
			},
			{
				Order:       2,
				Action:      "map_to_sap_gl",
				Description: "Map trade data to SAP GL journal entries",
				Parameters: map[string]interface{}{
					"target": "SAP_GL",
				},
				Assertions: []Assertion{
					{Type: "exists", Field: "journal_entry_id", Description: "Journal entry must be created"},
				},
			},
			{
				Order:       3,
				Action:      "verify_consistency",
				Description: "Verify data consistency between systems",
				Parameters: map[string]interface{}{
					"source": "Murex",
					"target": "SAP_GL",
				},
				Assertions: []Assertion{
					{Type: "equals", Field: "amount_match", ExpectedValue: true, Description: "Amounts must match"},
				},
			},
		},
		ExpectedResults: map[string]interface{}{
			"integration_status": "success",
			"records_synced":     1000,
		},
		DataPatterns: patterns,
		Priority:     "high",
		Tags:         []string{"integration", "murex", "sap-gl"},
		CreatedAt:    time.Now(),
	}

	return scenario, nil
}

// generatePerformanceScenario generates a performance test scenario.
func (tsg *TestScenarioGenerator) generatePerformanceScenario(id string, patterns []DataPattern) (*TestScenario, error) {
	scenario := &TestScenario{
		ID:          id,
		Name:        "Performance Load Test",
		Type:        "performance",
		Description: "Tests system performance under load",
		Steps: []ScenarioStep{
			{
				Order:       1,
				Action:      "setup_load",
				Description: "Setup load test environment",
				Parameters: map[string]interface{}{
					"concurrent_users": 100,
					"duration":         "5m",
				},
			},
			{
				Order:       2,
				Action:      "execute_queries",
				Description: "Execute knowledge graph queries under load",
				Parameters: map[string]interface{}{
					"query_count": 1000,
				},
				Assertions: []Assertion{
					{Type: "less_than", Field: "avg_latency_ms", ExpectedValue: 500, Description: "Average latency must be < 500ms"},
				},
			},
		},
		ExpectedResults: map[string]interface{}{
			"throughput":    200,
			"avg_latency_ms": 250,
		},
		DataPatterns: patterns,
		Priority:     "medium",
		Tags:         []string{"performance", "load-test"},
		CreatedAt:    time.Now(),
	}

	return scenario, nil
}

// generateEdgeCaseScenario generates an edge case test scenario.
func (tsg *TestScenarioGenerator) generateEdgeCaseScenario(id string, patterns []DataPattern) (*TestScenario, error) {
	scenario := &TestScenario{
		ID:          id,
		Name:        "Edge Case Test",
		Type:        "edge_case",
		Description: "Tests edge cases and boundary conditions",
		Steps: []ScenarioStep{
			{
				Order:       1,
				Action:      "test_null_values",
				Description: "Test handling of null values",
				Parameters: map[string]interface{}{
					"fields": []string{"counterparty_id", "trade_date"},
				},
				Assertions: []Assertion{
					{Type: "equals", Field: "error_handled", ExpectedValue: true, Description: "Null values must be handled gracefully"},
				},
			},
			{
				Order:       2,
				Action:      "test_boundary_values",
				Description: "Test boundary value conditions",
				Parameters: map[string]interface{}{
					"min_value": 0,
					"max_value": 999999999,
				},
			},
		},
		ExpectedResults: map[string]interface{}{
			"edge_cases_handled": true,
		},
		DataPatterns: patterns,
		Priority:     "medium",
		Tags:         []string{"edge-case", "boundary"},
		CreatedAt:    time.Now(),
	}

	return scenario, nil
}

// GetStats returns generation statistics.
func (tsg *TestScenarioGenerator) GetStats() GeneratorStats {
	return tsg.stats
}

