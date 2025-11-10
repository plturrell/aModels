package testing

import (
	"context"
	"fmt"
	"log"
	"time"
)

// ScenarioBuilder builds test scenarios from knowledge graph and process definitions.
type ScenarioBuilder struct {
	generator     *SampleGenerator
	logger        *log.Logger
	localaiClient *LocalAIClient
}

// NewScenarioBuilder creates a new scenario builder.
func NewScenarioBuilder(generator *SampleGenerator, localaiClient *LocalAIClient, logger *log.Logger) *ScenarioBuilder {
	return &ScenarioBuilder{
		generator:     generator,
		logger:        logger,
		localaiClient: localaiClient,
	}
}

// BuildScenarioFromPetriNet builds a test scenario from a Petri net.
func (sb *ScenarioBuilder) BuildScenarioFromPetriNet(ctx context.Context, petriNetID string) (*TestScenario, error) {
	sb.logger.Printf("Building test scenario from Petri net: %s", petriNetID)
	
	// Query Petri net from knowledge graph
	query := fmt.Sprintf(`
	MATCH (pn:Node)-[:HAS_TRANSITION]->(t:Node)
	WHERE pn.type = 'petri_net' AND pn.id = '%s'
	RETURN t.label as transition, t.properties_json as properties
	`, petriNetID)
	
	transitions, err := sb.generator.extractClient.QueryKnowledgeGraph(query, nil)
	if err != nil {
		return nil, fmt.Errorf("query Petri net transitions: %w", err)
	}
	
	scenario := &TestScenario{
		ID:          fmt.Sprintf("scenario_%s_%d", petriNetID, time.Now().Unix()),
		Name:        fmt.Sprintf("Test Scenario: %s", petriNetID),
		Description: fmt.Sprintf("Auto-generated from Petri net %s", petriNetID),
		Tables:      []*TableTestConfig{},
		Processes:   []*ProcessTestConfig{},
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	
	// Extract input/output tables from transitions
	inputTables := make(map[string]bool)
	outputTables := make(map[string]bool)
	
	for _, transition := range transitions {
		// props := parseProperties(transition["properties"]) // Unused in this scope
		
		processConfig := &ProcessTestConfig{
			ProcessID:    getString(transition, "transition", ""),
			ProcessType:  "controlm", // From Petri net
			InputTables:  []string{},
			OutputTables: []string{},
			ExpectedRows: make(map[string]int),
		}
		
		// Extract SQL queries from subprocesses
		query := fmt.Sprintf(`
		MATCH (t:Node)-[:HAS_SUBPROCESS]->(s:Node)
		WHERE t.type = 'petri_transition' AND t.label = '%s'
		  AND s.type = 'petri_subprocess' AND s.properties_json.subprocess_type = 'sql'
		RETURN s.properties_json.content as sql_query
		`, processConfig.ProcessID)
		
		sqlQueries, err := sb.generator.extractClient.QueryKnowledgeGraph(query, nil)
		if err == nil {
			for _, sqlData := range sqlQueries {
				if sqlQuery, ok := sqlData["sql_query"].(string); ok {
					processConfig.ValidationSQL = append(processConfig.ValidationSQL, sqlQuery)
					
					// Extract tables from SQL (simplified - would need proper SQL parsing)
					// This is a placeholder - would use SQL parser to extract tables
				}
			}
		}
		
		scenario.Processes = append(scenario.Processes, processConfig)
	}
	
	// Build table configs for input tables
	for tableName := range inputTables {
		schema, exists := sb.generator.knowledgeGraph.Tables[tableName]
		if exists {
			tableConfig := &TableTestConfig{
				TableName:    tableName,
				RowCount:     sb.determineRowCount(schema),
				QualityRules: sb.buildQualityRules(schema),
			}
			scenario.Tables = append(scenario.Tables, tableConfig)
		}
	}
	
	// Build table configs for output tables
	for tableName := range outputTables {
		schema, exists := sb.generator.knowledgeGraph.Tables[tableName]
		if exists {
			tableConfig := &TableTestConfig{
				TableName:    tableName,
				RowCount:     0, // Output tables shouldn't have initial data
				QualityRules: sb.buildQualityRules(schema),
			}
			scenario.Tables = append(scenario.Tables, tableConfig)
		}
	}
	
	return scenario, nil
}

// BuildScenarioFromProcessSequence builds a test scenario from table processing sequences.
func (sb *ScenarioBuilder) BuildScenarioFromProcessSequence(ctx context.Context) (*TestScenario, error) {
	sb.logger.Printf("Building test scenario from process sequences")
	
	// Query processing sequences from knowledge graph
	query := `
	MATCH (a:Node)-[r:RELATIONSHIP]->(b:Node)
	WHERE r.label = 'PROCESSES_BEFORE'
	RETURN a.label as source_table, b.label as target_table, r.properties_json as properties
	ORDER BY r.properties_json.sequence_order
	`
	
	sequences, err := sb.generator.extractClient.QueryKnowledgeGraph(query, nil)
	if err != nil {
		return nil, fmt.Errorf("query process sequences: %w", err)
	}
	
	scenario := &TestScenario{
		ID:          fmt.Sprintf("scenario_sequence_%d", time.Now().Unix()),
		Name:        "Test Scenario: Process Sequence",
		Description: "Auto-generated from table processing sequences",
		Tables:      []*TableTestConfig{},
		Processes:   []*ProcessTestConfig{},
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	
	// Build process configs from sequences
	processMap := make(map[string]*ProcessTestConfig)
	
	for i, seq := range sequences {
		sourceTable := getString(seq, "source_table", "")
		targetTable := getString(seq, "target_table", "")
		props := parseProperties(seq["properties"])
		
		processID := fmt.Sprintf("process_%d", i)
		processConfig := &ProcessTestConfig{
			ProcessID:    processID,
			ProcessType:  getString(props, "sequence_type", "sql"),
			InputTables:  []string{sourceTable},
			OutputTables: []string{targetTable},
			ExpectedRows: map[string]int{targetTable: 100}, // Default expected rows
		}
		
		processMap[processID] = processConfig
		scenario.Processes = append(scenario.Processes, processConfig)
		
		// Add table configs
		if schema, exists := sb.generator.knowledgeGraph.Tables[sourceTable]; exists {
			tableConfig := &TableTestConfig{
				TableName:    sourceTable,
				RowCount:     sb.determineRowCount(schema),
				QualityRules: sb.buildQualityRules(schema),
			}
			scenario.Tables = append(scenario.Tables, tableConfig)
		}
	}
	
	return scenario, nil
}

// determineRowCount determines appropriate row count based on table type.
// Uses configuration defaults if available, otherwise uses hardcoded defaults.
func (sb *ScenarioBuilder) determineRowCount(schema *TableSchema) int {
	// Note: This would ideally use Config, but to avoid circular dependencies,
	// we use hardcoded defaults. Config can be passed in if needed.
	switch schema.Type {
	case "reference":
		return 50 // Reference tables typically have fewer rows
	case "transaction":
		return 1000 // Transaction tables can have many rows
	case "staging":
		return 500 // Staging tables typically have moderate rows
	default:
		return 100
	}
}

// buildQualityRules builds quality rules from table schema.
func (sb *ScenarioBuilder) buildQualityRules(schema *TableSchema) []QualityRule {
	// Try LocalAI first if enabled
	if sb.localaiClient != nil && sb.localaiClient.IsEnabled() {
		ctx := context.Background()
		rules, err := sb.localaiClient.GenerateQualityRules(ctx, schema)
		if err == nil && len(rules) > 0 {
			sb.logger.Printf("Generated %d quality rules using LocalAI for table %s", len(rules), schema.Name)
			// Merge with basic rules
			basicRules := sb.buildBasicQualityRules(schema)
			return append(basicRules, rules...)
		}
		sb.logger.Printf("LocalAI quality rule generation failed, falling back to basic rules: %v", err)
	}
	
	// Fallback to basic rules
	return sb.buildBasicQualityRules(schema)
}

// buildBasicQualityRules builds basic quality rules from table schema.
func (sb *ScenarioBuilder) buildBasicQualityRules(schema *TableSchema) []QualityRule {
	rules := []QualityRule{}
	
	// Check for non-nullable columns
	for _, column := range schema.Columns {
		if !column.Nullable {
			rules = append(rules, QualityRule{
				Name:     fmt.Sprintf("non_null_%s", column.Name),
				Type:     "constraint",
				Rule:     fmt.Sprintf("%s IS NOT NULL", column.Name),
				Severity: "error",
			})
		}
	}
	
	// Check for foreign key constraints
	for _, fk := range schema.ForeignKeys {
		rules = append(rules, QualityRule{
			Name:     fmt.Sprintf("fk_%s", fk.Column),
			Type:     "relationship",
			Rule:     fmt.Sprintf("%s IN (SELECT %s FROM %s)", fk.Column, fk.ReferencedColumn, fk.ReferencedTable),
			Severity: "error",
		})
	}
	
	return rules
}

