package testing

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"
)

// SampleGenerator generates dynamic test data based on knowledge graph and database schema.
type SampleGenerator struct {
	db              *sql.DB
	logger          *log.Logger
	extractClient   ExtractClient // Interface for querying Extract service
	knowledgeGraph  *KnowledgeGraphCache
}

// ExtractClient interface for querying Extract service.
type ExtractClient interface {
	QueryKnowledgeGraph(query string, params map[string]any) ([]map[string]any, error)
	GetKnowledgeGraph(projectID, systemID string) (*GraphData, error)
}

// KnowledgeGraphCache caches knowledge graph data for performance.
type KnowledgeGraphCache struct {
	Tables      map[string]*TableSchema
	Columns     map[string]*ColumnSchema
	Relationships map[string][]*Relationship
	LastUpdated time.Time
}

// TableSchema represents a table's schema.
type TableSchema struct {
	Name        string
	Type        string // transaction, reference, staging, test
	Columns     []*ColumnSchema
	PrimaryKeys []string
	ForeignKeys []*ForeignKey
	Indexes     []string
	Constraints map[string]any
}

// ColumnSchema represents a column's schema.
type ColumnSchema struct {
	Name         string
	Type         string
	Nullable     bool
	DefaultValue any
	MaxLength    int
	Precision    int
	Scale        int
	IsPrimaryKey bool
	IsForeignKey bool
	References   *ForeignKey
	Patterns     []string // Learned patterns for value generation
	EnumValues   []string // For enum-like columns
}

// ForeignKey represents a foreign key relationship.
type ForeignKey struct {
	Column       string
	ReferencedTable string
	ReferencedColumn string
}

// Relationship represents a relationship between tables.
type Relationship struct {
	Type        string // PROCESSES_BEFORE, REFERENCES, etc.
	SourceTable string
	TargetTable string
	Properties  map[string]any
}

// GraphData represents knowledge graph data.
type GraphData struct {
	Nodes []map[string]any
	Edges []map[string]any
}

// TestScenario represents a test scenario.
type TestScenario struct {
	ID          string
	Name        string
	Description string
	Tables      []*TableTestConfig
	Processes   []*ProcessTestConfig
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// TableTestConfig configures test data generation for a table.
type TableTestConfig struct {
	TableName      string
	RowCount       int
	SeedData       map[string][]any // Column -> seed values
	Constraints    map[string]any
	QualityRules   []QualityRule
}

// ProcessTestConfig configures end-to-end process testing.
type ProcessTestConfig struct {
	ProcessID     string
	ProcessType   string // controlm, sql, workflow
	InputTables   []string
	OutputTables  []string
	ExpectedRows  map[string]int // table -> expected row count
	ValidationSQL []string
}

// QualityRule defines a data quality check.
type QualityRule struct {
	Name        string
	Type        string // constraint, pattern, relationship
	Rule        string
	Severity    string // error, warning, info
}

// TestExecution represents a test execution.
type TestExecution struct {
	ID            string
	ScenarioID    string
	Status        string // running, completed, failed
	StartTime     time.Time
	EndTime       time.Time
	Metrics       *ExecutionMetrics
	QualityIssues []QualityIssue
	Results       map[string]any
}

// ExecutionMetrics captures telemetry and latency metrics.
type ExecutionMetrics struct {
	TotalDuration    time.Duration
	ProcessDurations map[string]time.Duration // process -> duration
	DataVolumes      map[string]int            // table -> row count
	QueryLatencies   map[string]time.Duration  // query -> latency
	MemoryUsage      int64                     // bytes
	CPUUsage         float64
	Errors           []ErrorMetric
	Warnings         []string
}

// ErrorMetric represents an error with context.
type ErrorMetric struct {
	Timestamp   time.Time
	Process     string
	Error       string
	Context     map[string]any
	Severity    string
}

// QualityIssue represents a data or code quality issue.
type QualityIssue struct {
	Type        string // data_quality, code_quality, performance
	Severity    string // critical, high, medium, low
	Table       string
	Column      string
	Issue       string
	Details     map[string]any
	DetectedAt  time.Time
}

// NewSampleGenerator creates a new sample generator.
func NewSampleGenerator(db *sql.DB, extractClient ExtractClient, logger *log.Logger) *SampleGenerator {
	return &SampleGenerator{
		db:             db,
		logger:         logger,
		extractClient:  extractClient,
		knowledgeGraph: &KnowledgeGraphCache{
			Tables:        make(map[string]*TableSchema),
			Columns:       make(map[string]*ColumnSchema),
			Relationships: make(map[string][]*Relationship),
		},
	}
}

// LoadKnowledgeGraph loads knowledge graph data from Extract service.
func (sg *SampleGenerator) LoadKnowledgeGraph(ctx context.Context, projectID, systemID string) error {
	sg.logger.Printf("Loading knowledge graph for project=%s, system=%s", projectID, systemID)
	
	// Query for all tables
	tablesQuery := `
	MATCH (n:Node)
	WHERE n.type = 'table'
	RETURN n.id as id, n.label as label, n.properties_json as properties
	`
	
	tables, err := sg.extractClient.QueryKnowledgeGraph(tablesQuery, nil)
	if err != nil {
		return fmt.Errorf("query tables: %w", err)
	}
	
	// Load table schemas
	for _, tableData := range tables {
		tableName := tableData["label"].(string)
		props := parseProperties(tableData["properties"])
		
		schema := &TableSchema{
			Name:        tableName,
			Type:        getString(props, "table_classification", "unknown"),
			Columns:     []*ColumnSchema{},
			PrimaryKeys: []string{},
			ForeignKeys: []*ForeignKey{},
			Constraints: props,
		}
		
		// Load columns for this table
		columnsQuery := fmt.Sprintf(`
		MATCH (t:Node)-[:HAS_COLUMN]->(c:Node)
		WHERE t.type = 'table' AND t.label = '%s' AND c.type = 'column'
		RETURN c.id as id, c.label as label, c.properties_json as properties
		`, tableName)
		
		columns, err := sg.extractClient.QueryKnowledgeGraph(columnsQuery, nil)
		if err == nil {
			for _, colData := range columns {
				colName := colData["label"].(string)
				colProps := parseProperties(colData["properties"])
				
				column := &ColumnSchema{
					Name:         colName,
					Type:         getString(colProps, "type", "string"),
					Nullable:     getBool(colProps, "nullable", true),
					DefaultValue: colProps["default_value"],
					MaxLength:    getInt(colProps, "max_length", 0),
					Patterns:     getStringSlice(colProps, "patterns", []string{}),
					EnumValues:   getStringSlice(colProps, "enum_values", []string{}),
				}
				
				schema.Columns = append(schema.Columns, column)
				sg.knowledgeGraph.Columns[fmt.Sprintf("%s.%s", tableName, colName)] = column
			}
		}
		
		// Load relationships
		relationshipsQuery := fmt.Sprintf(`
		MATCH (t:Node)-[r:RELATIONSHIP]->(target:Node)
		WHERE t.type = 'table' AND t.label = '%s'
		RETURN r.label as type, target.label as target_table, r.properties_json as properties
		`, tableName)
		
		relationships, err := sg.extractClient.QueryKnowledgeGraph(relationshipsQuery, nil)
		if err == nil {
			for _, relData := range relationships {
				rel := &Relationship{
					Type:        getString(relData, "type", ""),
					SourceTable: tableName,
					TargetTable: getString(relData, "target_table", ""),
					Properties:  parseProperties(relData["properties"]),
				}
				sg.knowledgeGraph.Relationships[tableName] = append(
					sg.knowledgeGraph.Relationships[tableName],
					rel,
				)
			}
		}
		
		sg.knowledgeGraph.Tables[tableName] = schema
	}
	
	sg.knowledgeGraph.LastUpdated = time.Now()
	sg.logger.Printf("Loaded %d tables, %d columns from knowledge graph", 
		len(sg.knowledgeGraph.Tables), len(sg.knowledgeGraph.Columns))
	
	return nil
}

// GenerateSampleData generates test data for tables based on schema.
func (sg *SampleGenerator) GenerateSampleData(ctx context.Context, config *TableTestConfig) ([]map[string]any, error) {
	schema, exists := sg.knowledgeGraph.Tables[config.TableName]
	if !exists {
		return nil, fmt.Errorf("table %s not found in knowledge graph", config.TableName)
	}
	
	sg.logger.Printf("Generating %d rows for table %s", config.RowCount, config.TableName)
	
	rows := make([]map[string]any, 0, config.RowCount)
	rand.Seed(time.Now().UnixNano())
	
	// Generate reference data first if this is a reference table
	if schema.Type == "reference" {
		return sg.generateReferenceData(ctx, schema, config)
	}
	
	// Generate transaction/staging data
	for i := 0; i < config.RowCount; i++ {
		row := make(map[string]any)
		
		for _, column := range schema.Columns {
			value := sg.generateValueForColumn(ctx, column, schema, config, i)
			row[column.Name] = value
		}
		
		rows = append(rows, row)
	}
	
	return rows, nil
}

// generateReferenceData generates reference/lookup table data.
func (sg *SampleGenerator) generateReferenceData(ctx context.Context, schema *TableSchema, config *TableTestConfig) ([]map[string]any, error) {
	rows := make([]map[string]any, 0)
	
	// For reference tables, generate distinct values
	// Use enum values if available, or generate based on patterns
	
	for i := 0; i < config.RowCount; i++ {
		row := make(map[string]any)
		
		for _, column := range schema.Columns {
			// Use seed data if provided
			if seedValues, exists := config.SeedData[column.Name]; exists && len(seedValues) > 0 {
				row[column.Name] = seedValues[i%len(seedValues)]
			} else if len(column.EnumValues) > 0 {
				row[column.Name] = column.EnumValues[i%len(column.EnumValues)]
			} else {
				row[column.Name] = sg.generateValueForColumn(ctx, column, schema, config, i)
			}
		}
		
		rows = append(rows, row)
	}
	
	return rows, nil
}

// generateValueForColumn generates a value for a specific column.
func (sg *SampleGenerator) generateValueForColumn(
	ctx context.Context,
	column *ColumnSchema,
	schema *TableSchema,
	config *TableTestConfig,
	rowIndex int,
) any {
	// Use seed data if provided
	if seedValues, exists := config.SeedData[column.Name]; exists && len(seedValues) > 0 {
		return seedValues[rowIndex%len(seedValues)]
	}
	
	// Use enum values if available
	if len(column.EnumValues) > 0 {
		return column.EnumValues[rowIndex%len(column.EnumValues)]
	}
	
	// Use learned patterns if available
	if len(column.Patterns) > 0 {
		pattern := column.Patterns[rowIndex%len(column.Patterns)]
		return sg.generateFromPattern(pattern, column.Type)
	}
	
	// Generate based on column type
	switch strings.ToLower(column.Type) {
	case "string", "varchar", "text", "char":
		if column.MaxLength > 0 {
			return sg.generateString(column.MaxLength, column.Name)
		}
		return sg.generateString(50, column.Name)
	
	case "int", "integer", "bigint":
		return rand.Intn(1000000)
	
	case "decimal", "numeric", "float", "double":
		if column.Precision > 0 {
			return rand.Float64() * float64(column.Precision)
		}
		return rand.Float64() * 1000.0
	
	case "boolean", "bool":
		return rand.Intn(2) == 1
	
	case "date":
		return time.Now().AddDate(0, 0, -rand.Intn(365)).Format("2006-01-02")
	
	case "timestamp", "datetime":
		return time.Now().Add(-time.Duration(rand.Intn(86400)) * time.Second).Format(time.RFC3339)
	
	case "uuid":
		return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
			rand.Uint32(), rand.Uint32()&0xffff, rand.Uint32()&0xffff,
			rand.Uint32()&0xffff, rand.Uint64()&0xffffffffffff)
	
	default:
		return fmt.Sprintf("value_%d", rowIndex)
	}
}

// generateString generates a string value with intelligence.
func (sg *SampleGenerator) generateString(maxLength int, columnName string) string {
	// Use column name patterns to generate realistic data
	colLower := strings.ToLower(columnName)
	
	var value string
	if strings.Contains(colLower, "email") {
		value = fmt.Sprintf("user%d@example.com", rand.Intn(1000))
	} else if strings.Contains(colLower, "name") {
		value = fmt.Sprintf("Name_%d", rand.Intn(1000))
	} else if strings.Contains(colLower, "code") || strings.Contains(colLower, "id") {
		value = fmt.Sprintf("CODE_%04d", rand.Intn(10000))
	} else if strings.Contains(colLower, "status") {
		statuses := []string{"ACTIVE", "INACTIVE", "PENDING", "COMPLETED"}
		value = statuses[rand.Intn(len(statuses))]
	} else {
		value = fmt.Sprintf("value_%d", rand.Intn(10000))
	}
	
	if maxLength > 0 && len(value) > maxLength {
		return value[:maxLength]
	}
	return value
}

// generateFromPattern generates a value from a learned pattern.
func (sg *SampleGenerator) generateFromPattern(pattern, columnType string) any {
	// Parse pattern and generate accordingly
	// This would use pattern learning results
	return fmt.Sprintf("pattern_%s", pattern)
}

// ExecuteTestScenario executes an end-to-end test scenario.
func (sg *SampleGenerator) ExecuteTestScenario(ctx context.Context, scenario *TestScenario) (*TestExecution, error) {
	execution := &TestExecution{
		ID:         fmt.Sprintf("test_%d", time.Now().Unix()),
		ScenarioID: scenario.ID,
		Status:     "running",
		StartTime:  time.Now(),
		Metrics: &ExecutionMetrics{
			ProcessDurations: make(map[string]time.Duration),
			DataVolumes:      make(map[string]int),
			QueryLatencies:   make(map[string]time.Duration),
			Errors:           []ErrorMetric{},
		},
		QualityIssues: []QualityIssue{},
		Results:       make(map[string]any),
	}
	
	defer func() {
		execution.EndTime = time.Now()
		execution.Metrics.TotalDuration = execution.EndTime.Sub(execution.StartTime)
		if execution.Status == "running" {
			execution.Status = "completed"
		}
	}()
	
	sg.logger.Printf("Executing test scenario: %s", scenario.Name)
	
	// Step 1: Generate sample data for input tables
	for _, tableConfig := range scenario.Tables {
		startTime := time.Now()
		
		sampleData, err := sg.GenerateSampleData(ctx, tableConfig)
		if err != nil {
			execution.Metrics.Errors = append(execution.Metrics.Errors, ErrorMetric{
				Timestamp: time.Now(),
				Process:   fmt.Sprintf("generate_%s", tableConfig.TableName),
				Error:     err.Error(),
				Severity:  "error",
			})
			execution.Status = "failed"
			return execution, err
		}
		
		// Insert data into database
		if err := sg.insertData(ctx, tableConfig.TableName, sampleData); err != nil {
			execution.Metrics.Errors = append(execution.Metrics.Errors, ErrorMetric{
				Timestamp: time.Now(),
				Process:   fmt.Sprintf("insert_%s", tableConfig.TableName),
				Error:     err.Error(),
				Severity:  "error",
			})
			execution.Status = "failed"
			return execution, err
		}
		
		duration := time.Since(startTime)
		execution.Metrics.ProcessDurations[fmt.Sprintf("generate_%s", tableConfig.TableName)] = duration
		execution.Metrics.DataVolumes[tableConfig.TableName] = len(sampleData)
		execution.Results[tableConfig.TableName] = map[string]any{
			"rows_generated": len(sampleData),
			"duration":       duration,
		}
	}
	
	// Step 2: Execute processes
	for _, processConfig := range scenario.Processes {
		startTime := time.Now()
		
		if err := sg.executeProcess(ctx, processConfig, execution); err != nil {
			execution.Metrics.Errors = append(execution.Metrics.Errors, ErrorMetric{
				Timestamp: time.Now(),
				Process:   processConfig.ProcessID,
				Error:     err.Error(),
				Severity:  "error",
			})
			execution.Status = "failed"
			return execution, err
		}
		
		duration := time.Since(startTime)
		execution.Metrics.ProcessDurations[processConfig.ProcessID] = duration
	}
	
	// Step 3: Validate outputs
	if err := sg.validateOutputs(ctx, scenario, execution); err != nil {
		execution.Metrics.Errors = append(execution.Metrics.Errors, ErrorMetric{
			Timestamp: time.Now(),
			Process:   "validation",
			Error:     err.Error(),
			Severity:  "error",
		})
	}
	
	// Step 4: Check data quality
	sg.checkDataQuality(ctx, scenario, execution)
	
	// Step 5: Store execution results
	if err := sg.storeExecution(ctx, execution); err != nil {
		sg.logger.Printf("Warning: failed to store execution: %v", err)
	}
	
	return execution, nil
}

// insertData inserts generated data into database.
func (sg *SampleGenerator) insertData(ctx context.Context, tableName string, data []map[string]any) error {
	if len(data) == 0 {
		return nil
	}
	
	// Get column names from first row
	columns := make([]string, 0, len(data[0]))
	for col := range data[0] {
		columns = append(columns, col)
	}
	
	// Build INSERT statement
	placeholders := strings.Repeat("?,", len(columns))
	placeholders = placeholders[:len(placeholders)-1] // Remove trailing comma
	
	query := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)",
		tableName, strings.Join(columns, ","), placeholders)
	
	// Execute batch insert
	for _, row := range data {
		values := make([]any, 0, len(columns))
		for _, col := range columns {
			values = append(values, row[col])
		}
		
		if _, err := sg.db.ExecContext(ctx, query, values...); err != nil {
			return fmt.Errorf("insert row into %s: %w", tableName, err)
		}
	}
	
	return nil
}

// executeProcess executes a process (SQL, Control-M job, etc.).
func (sg *SampleGenerator) executeProcess(ctx context.Context, config *ProcessTestConfig, execution *TestExecution) error {
	startTime := time.Now()
	
	switch config.ProcessType {
	case "sql":
		// Execute SQL queries
		for _, sqlQuery := range config.ValidationSQL {
			queryStart := time.Now()
			if _, err := sg.db.ExecContext(ctx, sqlQuery); err != nil {
				return fmt.Errorf("execute SQL: %w", err)
			}
			execution.Metrics.QueryLatencies[sqlQuery] = time.Since(queryStart)
		}
	
	case "controlm":
		// Execute Control-M job (would call Control-M API)
		// For now, simulate execution
		time.Sleep(100 * time.Millisecond)
	
	case "workflow":
		// Execute workflow (would call workflow service)
		// For now, simulate execution
		time.Sleep(100 * time.Millisecond)
	}
	
	execution.Metrics.ProcessDurations[config.ProcessID] = time.Since(startTime)
	return nil
}

// validateOutputs validates process outputs.
func (sg *SampleGenerator) validateOutputs(ctx context.Context, scenario *TestScenario, execution *TestExecution) error {
	for _, processConfig := range scenario.Processes {
		for tableName, expectedRows := range processConfig.ExpectedRows {
			var actualRows int
			query := fmt.Sprintf("SELECT COUNT(*) FROM %s", tableName)
			if err := sg.db.QueryRowContext(ctx, query).Scan(&actualRows); err != nil {
				return fmt.Errorf("validate %s: %w", tableName, err)
			}
			
			if actualRows != expectedRows {
				execution.QualityIssues = append(execution.QualityIssues, QualityIssue{
					Type:       "data_quality",
					Severity:   "high",
					Table:      tableName,
					Issue:      fmt.Sprintf("Expected %d rows, got %d", expectedRows, actualRows),
					DetectedAt: time.Now(),
				})
			}
			
			execution.Results[tableName] = map[string]any{
				"expected_rows": expectedRows,
				"actual_rows":   actualRows,
			}
		}
	}
	
	return nil
}

// checkDataQuality checks data quality issues.
func (sg *SampleGenerator) checkDataQuality(ctx context.Context, scenario *TestScenario, execution *TestExecution) {
	for _, tableConfig := range scenario.Tables {
		// Check constraints
		for _, rule := range tableConfig.QualityRules {
			if err := sg.checkQualityRule(ctx, tableConfig.TableName, rule); err != nil {
				execution.QualityIssues = append(execution.QualityIssues, QualityIssue{
					Type:       rule.Type,
					Severity:   rule.Severity,
					Table:      tableConfig.TableName,
					Issue:      fmt.Sprintf("Quality rule violation: %s", rule.Name),
					Details:    map[string]any{"rule": rule.Rule, "error": err.Error()},
					DetectedAt: time.Now(),
				})
			}
		}
		
		// Check for NULLs in non-nullable columns
		schema, exists := sg.knowledgeGraph.Tables[tableConfig.TableName]
		if exists {
			for _, column := range schema.Columns {
				if !column.Nullable {
					var nullCount int
					query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE %s IS NULL",
						tableConfig.TableName, column.Name)
					if err := sg.db.QueryRowContext(ctx, query).Scan(&nullCount); err == nil && nullCount > 0 {
						execution.QualityIssues = append(execution.QualityIssues, QualityIssue{
							Type:       "data_quality",
							Severity:   "high",
							Table:      tableConfig.TableName,
							Column:     column.Name,
							Issue:      fmt.Sprintf("Found %d NULL values in non-nullable column", nullCount),
							DetectedAt: time.Now(),
						})
					}
				}
			}
		}
	}
}

// checkQualityRule checks a specific quality rule.
func (sg *SampleGenerator) checkQualityRule(ctx context.Context, tableName string, rule QualityRule) error {
	// Execute rule as SQL query
	var result int
	query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE NOT (%s)", tableName, rule.Rule)
	if err := sg.db.QueryRowContext(ctx, query).Scan(&result); err != nil {
		return err
	}
	
	if result > 0 {
		return fmt.Errorf("quality rule violation: %d rows failed rule", result)
	}
	
	return nil
}

// storeExecution stores test execution results in database.
func (sg *SampleGenerator) storeExecution(ctx context.Context, execution *TestExecution) error {
	// Create test_executions table if it doesn't exist
	createTableSQL := `
	CREATE TABLE IF NOT EXISTS test_executions (
		id TEXT PRIMARY KEY,
		scenario_id TEXT,
		status TEXT,
		start_time TIMESTAMP,
		end_time TIMESTAMP,
		metrics_json TEXT,
		quality_issues_json TEXT,
		results_json TEXT
	)
	`
	if _, err := sg.db.ExecContext(ctx, createTableSQL); err != nil {
		return fmt.Errorf("create test_executions table: %w", err)
	}
	
	// Serialize metrics and issues
	metricsJSON, _ := json.Marshal(execution.Metrics)
	issuesJSON, _ := json.Marshal(execution.QualityIssues)
	resultsJSON, _ := json.Marshal(execution.Results)
	
	// Insert execution
	insertSQL := `
	INSERT INTO test_executions (id, scenario_id, status, start_time, end_time, metrics_json, quality_issues_json, results_json)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`
	_, err := sg.db.ExecContext(ctx, insertSQL,
		execution.ID,
		execution.ScenarioID,
		execution.Status,
		execution.StartTime,
		execution.EndTime,
		string(metricsJSON),
		string(issuesJSON),
		string(resultsJSON),
	)
	
	return err
}

// Helper functions

func parseProperties(props any) map[string]any {
	if props == nil {
		return make(map[string]any)
	}
	
	if propsMap, ok := props.(map[string]any); ok {
		return propsMap
	}
	
	if propsStr, ok := props.(string); ok {
		var result map[string]any
		if err := json.Unmarshal([]byte(propsStr), &result); err == nil {
			return result
		}
	}
	
	return make(map[string]any)
}

func getString(m map[string]any, key string, defaultValue string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return defaultValue
}

func getBool(m map[string]any, key string, defaultValue bool) bool {
	if val, ok := m[key].(bool); ok {
		return val
	}
	return defaultValue
}

func getInt(m map[string]any, key string, defaultValue int) int {
	if val, ok := m[key].(float64); ok {
		return int(val)
	}
	if val, ok := m[key].(int); ok {
		return val
	}
	return defaultValue
}

func getStringSlice(m map[string]any, key string, defaultValue []string) []string {
	if val, ok := m[key].([]string); ok {
		return val
	}
	if val, ok := m[key].([]any); ok {
		result := make([]string, 0, len(val))
		for _, v := range val {
			if str, ok := v.(string); ok {
				result = append(result, str)
			}
		}
		return result
	}
	return defaultValue
}

