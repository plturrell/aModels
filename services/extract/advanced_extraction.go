//go:build disabled
package main

import (
	stdctx "context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"
)

// AdvancedExtractionResult contains extracted advanced information.
type AdvancedExtractionResult struct {
	TableProcessSequences []TableProcessSequence `json:"table_process_sequences"`
	CodeParameters        []CodeParameter        `json:"code_parameters"`
	HardcodedLists        []HardcodedList        `json:"hardcoded_lists"`
	TableClassifications []TableClassification  `json:"table_classifications"`
	TestingEndpoints      []TestingEndpoint      `json:"testing_endpoints"`
}

// TableProcessSequence represents the sequence/order of table processing.
type TableProcessSequence struct {
	SequenceID   string   `json:"sequence_id"`
	Tables       []string `json:"tables"`        // Ordered list of tables
	SourceType   string   `json:"source_type"`   // sql, controlm, ddl, etc.
	SourceFile   string   `json:"source_file"`
	SequenceType string   `json:"sequence_type"` // insert, update, select, etc.
	Order        int      `json:"order"`         // Processing order
}

// CodeParameter represents a parameter found in code.
type CodeParameter struct {
	Name         string `json:"name"`
	Type         string `json:"type"`          // string, int, boolean, etc.
	Source       string `json:"source"`       // sql, controlm, json, etc.
	SourceFile   string `json:"source_file"`
	IsRequired   bool   `json:"is_required"`
	DefaultValue string `json:"default_value,omitempty"`
	Context      string `json:"context"`      // WHERE clause, JOIN condition, etc.
}

// HardcodedList represents a hardcoded list/constant found in code.
type HardcodedList struct {
	Name       string   `json:"name"`
	Values     []string `json:"values"`
	Source     string   `json:"source"`     // sql, controlm, json, etc.
	SourceFile string   `json:"source_file"`
	Type       string   `json:"type"`        // IN clause, enum, constant list, etc.
	Context    string   `json:"context"`
}

// TableClassification classifies a table as transaction or reference.
// Phase 5: Extended with Props for quality scores and review flags.
type TableClassification struct {
	TableName      string   `json:"table_name"`
	Classification string   `json:"classification"` // transaction, reference, lookup, staging, etc.
	Confidence     float64  `json:"confidence"`     // 0.0 to 1.0
	Evidence       []string `json:"evidence"`       // Reasons for classification
	Patterns       []string `json:"patterns"`       // Patterns that led to classification
	Props          map[string]any `json:"props,omitempty"` // Phase 5: Additional properties (quality_score, needs_review)
}

// TestingEndpoint represents a testing/test endpoint.
type TestingEndpoint struct {
	Endpoint      string   `json:"endpoint"`
	Method        string   `json:"method"`        // GET, POST, etc.
	Source        string   `json:"source"`        // api, controlm, etc.
	SourceFile    string   `json:"source_file"`
	IsTest        bool     `json:"is_test"`
	TestIndicators []string `json:"test_indicators"` // test, mock, stub, etc.
}

// AdvancedExtractor performs advanced extraction from parsed code.
type AdvancedExtractor struct {
	logger            *log.Logger
	terminologyLearner *TerminologyLearner // Phase 10: LNN-based terminology learning
}

// NewAdvancedExtractor creates a new advanced extractor.
func NewAdvancedExtractor(logger *log.Logger) *AdvancedExtractor {
	return &AdvancedExtractor{
		logger:            logger,
		terminologyLearner: nil, // Will be set via SetTerminologyLearner
	}
}

// SetTerminologyLearner sets the terminology learner (Phase 10).
func (ae *AdvancedExtractor) SetTerminologyLearner(learner *TerminologyLearner) {
	ae.terminologyLearner = learner
}

// ExtractAdvanced extracts advanced information from parsed data.
func (ae *AdvancedExtractor) ExtractAdvanced(
	sqlQueries []string,
	controlMFiles []string,
	ddlStatements []string,
	jsonTables []map[string]any,
) *AdvancedExtractionResult {
	result := &AdvancedExtractionResult{
		TableProcessSequences: []TableProcessSequence{},
		CodeParameters:        []CodeParameter{},
		HardcodedLists:         []HardcodedList{},
		TableClassifications:   []TableClassification{},
		TestingEndpoints:       []TestingEndpoint{},
	}

	// Extract table process sequences from SQL
	for i, sql := range sqlQueries {
		sequences := ae.extractTableSequencesFromSQL(sql, fmt.Sprintf("sql_%d", i))
		result.TableProcessSequences = append(result.TableProcessSequences, sequences...)
		
		params := ae.extractParametersFromSQL(sql, fmt.Sprintf("sql_%d", i))
		result.CodeParameters = append(result.CodeParameters, params...)
		
		lists := ae.extractHardcodedListsFromSQL(sql, fmt.Sprintf("sql_%d", i))
		result.HardcodedLists = append(result.HardcodedLists, lists...)
	}

	// Extract from Control-M files
	for i, file := range controlMFiles {
		sequences := ae.extractTableSequencesFromControlM(file, fmt.Sprintf("controlm_%d", i))
		result.TableProcessSequences = append(result.TableProcessSequences, sequences...)
		
		params := ae.extractParametersFromControlM(file, fmt.Sprintf("controlm_%d", i))
		result.CodeParameters = append(result.CodeParameters, params...)
		
		endpoints := ae.extractTestingEndpointsFromControlM(file, fmt.Sprintf("controlm_%d", i))
		result.TestingEndpoints = append(result.TestingEndpoints, endpoints...)
	}

	// Extract from DDL statements
	for i, ddl := range ddlStatements {
		classifications := ae.classifyTablesFromDDL(ddl, fmt.Sprintf("ddl_%d", i))
		result.TableClassifications = append(result.TableClassifications, classifications...)
	}

	// Extract from JSON tables
	for i, jsonTable := range jsonTables {
		classifications := ae.classifyTablesFromJSON(jsonTable, fmt.Sprintf("json_%d", i))
		result.TableClassifications = append(result.TableClassifications, classifications...)
		
		lists := ae.extractHardcodedListsFromJSON(jsonTable, fmt.Sprintf("json_%d", i))
		result.HardcodedLists = append(result.HardcodedLists, lists...)
	}

	// Deduplicate and merge classifications
	result.TableClassifications = ae.mergeTableClassifications(result.TableClassifications)

	return result
}

// extractTableSequencesFromSQL extracts table processing sequences from SQL queries.
func (ae *AdvancedExtractor) extractTableSequencesFromSQL(sql, sourceID string) []TableProcessSequence {
	sequences := []TableProcessSequence{}
	
	// Patterns to detect table processing order:
	// 1. INSERT INTO ... SELECT FROM ... (target, then source)
	// 2. UPDATE ... FROM ... (target, then source)
	// 3. SELECT ... FROM ... JOIN ... (left to right)
	// 4. CTEs (WITH ... AS ... SELECT) (sequential)
	
	// Detect INSERT INTO ... SELECT FROM pattern
	insertPattern := regexp.MustCompile(`(?i)INSERT\s+INTO\s+(\w+)\s+.*?SELECT\s+.*?FROM\s+(\w+)`)
	matches := insertPattern.FindAllStringSubmatch(sql, -1)
	for order, match := range matches {
		if len(match) >= 3 {
			sequences = append(sequences, TableProcessSequence{
				SequenceID:   fmt.Sprintf("%s_insert_%d", sourceID, order),
				Tables:       []string{match[2], match[1]}, // source, then target
				SourceType:   "sql",
				SourceFile:   sourceID,
				SequenceType: "insert",
				Order:        order,
			})
		}
	}
	
	// Detect UPDATE ... FROM pattern
	updatePattern := regexp.MustCompile(`(?i)UPDATE\s+(\w+)\s+.*?FROM\s+(\w+)`)
	matches = updatePattern.FindAllStringSubmatch(sql, -1)
	for order, match := range matches {
		if len(match) >= 3 {
			sequences = append(sequences, TableProcessSequence{
				SequenceID:   fmt.Sprintf("%s_update_%d", sourceID, order),
				Tables:       []string{match[2], match[1]}, // source, then target
				SourceType:   "sql",
				SourceFile:   sourceID,
				SequenceType: "update",
				Order:        order,
			})
		}
	}
	
	// Detect SELECT ... FROM ... JOIN pattern (left to right)
	selectPattern := regexp.MustCompile(`(?i)SELECT\s+.*?FROM\s+(\w+)(?:\s+JOIN\s+(\w+))*`)
	matches = selectPattern.FindAllStringSubmatch(sql, -1)
	for order, match := range matches {
		if len(match) >= 2 {
			tables := []string{match[1]}
			// Extract JOIN tables
			joinPattern := regexp.MustCompile(`(?i)JOIN\s+(\w+)`)
			joinMatches := joinPattern.FindAllStringSubmatch(sql, -1)
			for _, joinMatch := range joinMatches {
				if len(joinMatch) >= 2 {
					tables = append(tables, joinMatch[1])
				}
			}
			sequences = append(sequences, TableProcessSequence{
				SequenceID:   fmt.Sprintf("%s_select_%d", sourceID, order),
				Tables:       tables,
				SourceType:   "sql",
				SourceFile:   sourceID,
				SequenceType: "select",
				Order:        order,
			})
		}
	}
	
	// Detect CTE sequences (WITH ... AS ...)
	ctePattern := regexp.MustCompile(`(?i)WITH\s+(\w+)\s+AS\s+\([^)]+\)\s*,?\s*(\w+)\s+AS\s+\([^)]+\)`)
	matches = ctePattern.FindAllStringSubmatch(sql, -1)
	for order, match := range matches {
		if len(match) >= 3 {
			sequences = append(sequences, TableProcessSequence{
				SequenceID:   fmt.Sprintf("%s_cte_%d", sourceID, order),
				Tables:       []string{match[1], match[2]}, // CTE order
				SourceType:   "sql",
				SourceFile:   sourceID,
				SequenceType: "cte",
				Order:        order,
			})
		}
	}
	
	return sequences
}

// extractParametersFromSQL extracts parameters from SQL queries.
func (ae *AdvancedExtractor) extractParametersFromSQL(sql, sourceID string) []CodeParameter {
	params := []CodeParameter{}
	
	// Detect WHERE clause parameters
	wherePattern := regexp.MustCompile(`(?i)WHERE\s+(\w+)\s*=\s*(\?|:(\w+)|@(\w+))`)
	matches := wherePattern.FindAllStringSubmatch(sql, -1)
	for _, match := range matches {
		paramName := match[1] // column name
		if len(match) > 3 && match[3] != "" {
			paramName = match[3] // named parameter
		}
		if len(match) > 4 && match[4] != "" {
			paramName = match[4] // @parameter
		}
		
		params = append(params, CodeParameter{
			Name:       paramName,
			Type:       "unknown", // Would need type inference
			Source:     "sql",
			SourceFile: sourceID,
			IsRequired: true,
			Context:    "WHERE clause",
		})
	}
	
	// Detect function parameters
	funcPattern := regexp.MustCompile(`(?i)(\w+)\s*\([^)]*\)`)
	matches = funcPattern.FindAllStringSubmatch(sql, -1)
	for _, match := range matches {
		// Extract function name and parameters
		funcName := match[1]
		params = append(params, CodeParameter{
			Name:       funcName,
			Type:       "function",
			Source:     "sql",
			SourceFile: sourceID,
			Context:    "function call",
		})
	}
	
	return params
}

// extractHardcodedListsFromSQL extracts hardcoded lists from SQL (e.g., IN clauses).
func (ae *AdvancedExtractor) extractHardcodedListsFromSQL(sql, sourceID string) []HardcodedList {
	lists := []HardcodedList{}
	
	// Detect IN clause hardcoded lists
	inPattern := regexp.MustCompile(`(?i)IN\s*\(([^)]+)\)`)
	matches := inPattern.FindAllStringSubmatch(sql, -1)
	for i, match := range matches {
		if len(match) >= 2 {
			valuesStr := match[1]
			// Parse values (handle strings, numbers, etc.)
			values := parseValueList(valuesStr)
			
			if len(values) > 0 {
				lists = append(lists, HardcodedList{
					Name:       fmt.Sprintf("in_clause_%d", i),
					Values:     values,
					Source:     "sql",
					SourceFile: sourceID,
					Type:       "IN clause",
					Context:    "SQL IN clause",
				})
			}
		}
	}
	
	// Detect CASE WHEN ... THEN ... ELSE patterns (hardcoded logic)
	casePattern := regexp.MustCompile(`(?i)CASE\s+WHEN\s+([^T]+)\s+THEN\s+([^E]+)\s+ELSE\s+([^E]+)`)
	matches = casePattern.FindAllStringSubmatch(sql, -1)
	for i, match := range matches {
		if len(match) >= 4 {
			lists = append(lists, HardcodedList{
				Name:       fmt.Sprintf("case_when_%d", i),
				Values:     []string{match[2], match[3]}, // THEN and ELSE values
				Source:     "sql",
				SourceFile: sourceID,
				Type:       "CASE WHEN",
				Context:    "SQL CASE statement",
			})
		}
	}
	
	return lists
}

// classifyTablesFromDDL classifies tables as transaction vs reference based on DDL patterns.
func (ae *AdvancedExtractor) classifyTablesFromDDL(ddl, sourceID string) []TableClassification {
	classifications := []TableClassification{}
	
	// Extract table names from DDL
	createTablePattern := regexp.MustCompile(`(?i)CREATE\s+TABLE\s+(\w+)`)
	matches := createTablePattern.FindAllStringSubmatch(ddl, -1)
	
	for _, match := range matches {
		if len(match) >= 2 {
			tableName := match[1]
			classification := ae.classifyTable(tableName, ddl, "")
			classifications = append(classifications, classification)
		}
	}
	
	return classifications
}

// classifyTablesFromJSON classifies tables from JSON schema.
func (ae *AdvancedExtractor) classifyTablesFromJSON(jsonTable map[string]any, sourceID string) []TableClassification {
	classifications := []TableClassification{}
	
	// Extract table information from JSON
	for tableName, tableData := range jsonTable {
		if tableDataMap, ok := tableData.(map[string]any); ok {
			// Convert to string for classification
			tableStr, _ := json.Marshal(tableDataMap)
			classification := ae.classifyTable(tableName, string(tableStr), sourceID)
			classifications = append(classifications, classification)
		}
	}
	
	return classifications
}

// classifyTable classifies a single table as transaction, reference, etc.
// Uses sap-rpt-1-oss if available, otherwise falls back to pattern matching.
// Phase 5: Supports multi-task learning (classification + regression) and active learning
func (ae *AdvancedExtractor) classifyTable(tableName, context, sourceID string) TableClassification {
	// Try advanced multi-task SAP-RPT if enabled (Phase 5)
	if useAdvanced := os.Getenv("USE_SAP_RPT_ADVANCED") == "true"; useAdvanced {
		trainingDataPath := os.Getenv("SAP_RPT_TRAINING_DATA_PATH")
		if trainingDataPath != "" {
			metadata := map[string]any{} // Could extract from context
			classification, qualityScore, needsReview := ae.classifyTableWithAdvancedSAPRPT(
				tableName, context, sourceID, trainingDataPath, metadata)
			
			// Store quality score in classification
			if classification.Props == nil {
				classification.Props = make(map[string]any)
			}
			classification.Props["quality_score"] = qualityScore
			classification.Props["needs_review"] = needsReview
			
			if classification.Classification != "unknown" {
				return classification
			}
		}
	}
	
	// Try full SAP-RPT classifier if enabled and training data available (Phase 4)
	if useSAPRPTClassification := os.Getenv("USE_SAP_RPT_CLASSIFICATION"); useSAPRPTClassification == "true" {
		trainingDataPath := os.Getenv("SAP_RPT_TRAINING_DATA_PATH")
		if trainingDataPath != "" {
			if classification := ae.classifyTableWithFullSAPRPT(tableName, context, sourceID, trainingDataPath); classification.Classification != "unknown" {
				return classification
			}
		}
		
		// Fallback to feature-based classification
		if classification := ae.classifyTableWithSAPRPT(tableName, context, sourceID); classification.Classification != "unknown" {
			return classification
		}
		// Fallback to pattern matching if sap-rpt-1-oss fails
	}
	
	// Phase 10: Try LNN-based classification if available
	if ae.terminologyLearner != nil {
		ctx := stdctx.Background()
		domain, domainConf := ae.terminologyLearner.InferDomain(ctx, tableName, tableName, map[string]any{"context": context})
		
		// Map domain to table classification
		if domainConf > 0.6 {
			classification := TableClassification{
				TableName:      tableName,
				Classification: "unknown",
				Confidence:     domainConf,
				Evidence:       []string{fmt.Sprintf("LNN inferred domain: %s", domain)},
				Patterns:       []string{},
			}
			
			// Map domain to table type
			if domain == "financial" || domain == "order" {
				classification.Classification = "transaction"
			} else if domain == "customer" || domain == "product" {
				classification.Classification = "reference"
			}
			
			if classification.Classification != "unknown" {
				return classification
			}
		}
	}
	
	// Pattern-based classification (original implementation - fallback)
	classification := TableClassification{
		TableName:      tableName,
		Classification: "unknown",
		Confidence:     0.0,
		Evidence:       []string{},
		Patterns:       []string{},
	}
	
	tableLower := strings.ToLower(tableName)
	
	// Transaction table patterns
	transactionPatterns := []string{
		"trans", "txn", "tx", "order", "payment", "invoice", "receipt",
		"event", "log", "audit", "history", "fact", "measurement",
		"transaction", "settlement", "clearing",
	}
	
	// Reference/lookup table patterns
	referencePatterns := []string{
		"ref", "lookup", "code", "dict", "master", "config", "setting",
		"parameter", "type", "category", "status", "enum", "domain",
		"dimension", "dim", "dim_",
	}
	
	// Staging table patterns
	stagingPatterns := []string{
		"staging", "stage", "temp", "tmp", "intermediate", "landing",
		"raw", "source", "extract",
	}
	
	// Test table patterns
	testPatterns := []string{
		"test", "mock", "stub", "fake", "sample", "demo", "trial",
	}
	
	evidence := []string{}
	confidence := 0.0
	
	// Check transaction patterns
	for _, pattern := range transactionPatterns {
		if strings.Contains(tableLower, pattern) {
			evidence = append(evidence, fmt.Sprintf("Contains transaction pattern: %s", pattern))
			confidence += 0.3
			classification.Patterns = append(classification.Patterns, pattern)
		}
	}
	
	// Check reference patterns
	for _, pattern := range referencePatterns {
		if strings.Contains(tableLower, pattern) {
			evidence = append(evidence, fmt.Sprintf("Contains reference pattern: %s", pattern))
			confidence += 0.3
			classification.Patterns = append(classification.Patterns, pattern)
		}
	}
	
	// Check staging patterns
	for _, pattern := range stagingPatterns {
		if strings.Contains(tableLower, pattern) {
			evidence = append(evidence, fmt.Sprintf("Contains staging pattern: %s", pattern))
			confidence += 0.3
			classification.Patterns = append(classification.Patterns, pattern)
		}
	}
	
	// Check test patterns
	for _, pattern := range testPatterns {
		if strings.Contains(tableLower, pattern) {
			evidence = append(evidence, fmt.Sprintf("Contains test pattern: %s", pattern))
			confidence += 0.3
			classification.Patterns = append(classification.Patterns, pattern)
		}
	}
	
	// Analyze table structure from context
	if strings.Contains(strings.ToLower(context), "primary key") {
		// Reference tables often have simple primary keys
		if strings.Contains(strings.ToLower(context), "foreign key") {
			evidence = append(evidence, "Has foreign key relationships")
			confidence += 0.2
		}
	}
	
	// Determine final classification based on highest confidence
	if confidence >= 0.6 {
		if containsAny(tableLower, transactionPatterns) {
			classification.Classification = "transaction"
		} else if containsAny(tableLower, referencePatterns) {
			classification.Classification = "reference"
		} else if containsAny(tableLower, stagingPatterns) {
			classification.Classification = "staging"
		} else if containsAny(tableLower, testPatterns) {
			classification.Classification = "test"
		}
	} else {
		classification.Classification = "unknown"
	}
	
	classification.Confidence = confidence
	if classification.Confidence > 1.0 {
		classification.Confidence = 1.0
	}
	classification.Evidence = evidence
	
	return classification
}

// extractTableSequencesFromControlM extracts table processing sequences from Control-M files.
func (ae *AdvancedExtractor) extractTableSequencesFromControlM(controlMContent, sourceID string) []TableProcessSequence {
	sequences := []TableProcessSequence{}
	
	// Control-M files contain job dependencies which indicate processing order
	// Look for job dependencies and extract table references from job commands
	
	// Pattern: Job dependencies indicate sequence
	dependencyPattern := regexp.MustCompile(`(?i)(?:wait|depend|after)\s*[=:]\s*(\w+)`)
	matches := dependencyPattern.FindAllStringSubmatch(controlMContent, -1)
	
	// Extract table names from SQL commands in jobs
	tablePattern := regexp.MustCompile(`(?i)(?:table|from|into|update)\s+(\w+)`)
	tableMatches := tablePattern.FindAllStringSubmatch(controlMContent, -1)
	
	tables := []string{}
	for _, match := range tableMatches {
		if len(match) >= 2 {
			tableName := match[1]
			if !contains(tables, tableName) {
				tables = append(tables, tableName)
			}
		}
	}
	
	if len(tables) > 0 {
		sequences = append(sequences, TableProcessSequence{
			SequenceID:   fmt.Sprintf("%s_controlm", sourceID),
			Tables:       tables,
			SourceType:   "controlm",
			SourceFile:   sourceID,
			SequenceType: "controlm_job",
			Order:        0,
		})
	}
	
	return sequences
}

// extractParametersFromControlM extracts parameters from Control-M files.
func (ae *AdvancedExtractor) extractParametersFromControlM(controlMContent, sourceID string) []CodeParameter {
	params := []CodeParameter{}
	
	// Extract variables from Control-M
	varPattern := regexp.MustCompile(`(?i)(?:variable|var|param)\s*[=:]\s*(\w+)`)
	matches := varPattern.FindAllStringSubmatch(controlMContent, -1)
	
	for _, match := range matches {
		if len(match) >= 2 {
			params = append(params, CodeParameter{
				Name:       match[1],
				Type:       "string", // Control-M variables are typically strings
				Source:     "controlm",
				SourceFile: sourceID,
				IsRequired: false, // Control-M variables often have defaults
				Context:    "Control-M variable",
			})
		}
	}
	
	return params
}

// extractTestingEndpointsFromControlM extracts testing endpoints from Control-M files.
func (ae *AdvancedExtractor) extractTestingEndpointsFromControlM(controlMContent, sourceID string) []TestingEndpoint {
	endpoints := []TestingEndpoint{}
	
	// Look for test indicators in Control-M job names and descriptions
	testPattern := regexp.MustCompile(`(?i)(test|mock|stub|fake|sample|demo|trial)`)
	testMatches := testPattern.FindAllStringSubmatch(controlMContent, -1)
	
	// Extract endpoint patterns (HTTP URLs, API calls)
	endpointPattern := regexp.MustCompile(`(?i)(https?://[^\s]+|/[a-z0-9/_-]+)`)
	endpointMatches := endpointPattern.FindAllStringSubmatch(controlMContent, -1)
	
	for _, match := range endpointMatches {
		if len(match) >= 2 {
			endpoint := match[1]
			isTest := testPattern.MatchString(endpoint) || testPattern.MatchString(controlMContent)
			
			indicators := []string{}
			if isTest {
				indicators = append(indicators, "test indicator in content")
			}
			
			endpoints = append(endpoints, TestingEndpoint{
				Endpoint:      endpoint,
				Method:        "unknown", // Would need to parse HTTP method
				Source:        "controlm",
				SourceFile:    sourceID,
				IsTest:        isTest,
				TestIndicators: indicators,
			})
		}
	}
	
	return endpoints
}

// extractHardcodedListsFromJSON extracts hardcoded lists from JSON tables.
func (ae *AdvancedExtractor) extractHardcodedListsFromJSON(jsonTable map[string]any, sourceID string) []HardcodedList {
	lists := []HardcodedList{}
	
	// Look for enum-like structures in JSON
	for key, value := range jsonTable {
		if valueMap, ok := value.(map[string]any); ok {
			// Check for enum/constant patterns
			if enumValues, ok := valueMap["enum"].([]any); ok {
				values := []string{}
				for _, v := range enumValues {
					if str, ok := v.(string); ok {
						values = append(values, str)
					}
				}
				if len(values) > 0 {
					lists = append(lists, HardcodedList{
						Name:       key,
						Values:     values,
						Source:     "json",
						SourceFile: sourceID,
						Type:       "enum",
						Context:    "JSON enum definition",
					})
				}
			}
		}
	}
	
	return lists
}

// mergeTableClassifications merges duplicate table classifications.
func (ae *AdvancedExtractor) mergeTableClassifications(classifications []TableClassification) []TableClassification {
	// Group by table name
	tableMap := make(map[string]*TableClassification)
	
	for i := range classifications {
		tableName := classifications[i].TableName
		if existing, ok := tableMap[tableName]; ok {
			// Merge: take classification with higher confidence
			if classifications[i].Confidence > existing.Confidence {
				*existing = classifications[i]
			}
			// Merge evidence
			existing.Evidence = append(existing.Evidence, classifications[i].Evidence...)
			existing.Patterns = append(existing.Patterns, classifications[i].Patterns...)
		} else {
			tableMap[tableName] = &classifications[i]
		}
	}
	
	// Convert back to slice
	result := make([]TableClassification, 0, len(tableMap))
	for _, classification := range tableMap {
		result = append(result, *classification)
	}
	
	// Sort by table name
	sort.Slice(result, func(i, j int) bool {
		return result[i].TableName < result[j].TableName
	})
	
	return result
}

// classifyTableWithAdvancedSAPRPT calls the advanced multi-task script (Phase 5: classification + regression)
func (ae *AdvancedExtractor) classifyTableWithAdvancedSAPRPT(tableName, context, sourceID, trainingDataPath string, metadata map[string]any) (TableClassification, float64, bool) {
	columns := []map[string]any{}
	if strings.Contains(context, "CREATE TABLE") {
		columnPattern := regexp.MustCompile(`(?i)(\w+)\s+(\w+)`)
		matches := columnPattern.FindAllStringSubmatch(context, -1)
		for _, match := range matches {
			if len(match) >= 3 {
				columns = append(columns, map[string]any{
					"name": match[1],
					"type": match[2],
				})
			}
		}
	}

	columnsJSON, _ := json.Marshal(columns)
	metadataJSON, _ := json.Marshal(metadata)

	cmd := exec.Command("python3", "./scripts/sap_rpt_advanced.py",
		"--table-name", tableName,
		"--columns", string(columnsJSON),
		"--context", context,
		"--training-data", trainingDataPath,
		"--metadata", string(metadataJSON),
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			ae.logger.Printf("advanced sap-rpt prediction failed: %v, stderr: %s", err, string(exitErr.Stderr))
		}
		return TableClassification{
			TableName:      tableName,
			Classification: "unknown",
			Confidence:     0.0,
			Source:         sourceID,
		}, 0.0, false
	}

	var result map[string]any
	if err := json.Unmarshal(output, &result); err != nil {
		return TableClassification{
			TableName:      tableName,
			Classification: "unknown",
			Confidence:     0.0,
			Source:         sourceID,
		}, 0.0, false
	}

	classification := "unknown"
	if cls, ok := result["classification"].(string); ok {
		classification = cls
	}

	confidence := 0.0
	if conf, ok := result["classification_confidence"].(float64); ok {
		confidence = conf
	}

	qualityScore := 0.0
	if qs, ok := result["quality_score"].(float64); ok {
		qualityScore = qs
	}

	needsReview := false
	if nr, ok := result["needs_review"].(bool); ok {
		needsReview = nr
	}

	evidence := []string{}
	if ev, ok := result["uncertainty_reason"].(string); ok && ev != "" {
		evidence = append(evidence, ev)
	}
	if method, ok := result["method"].(string); ok {
		evidence = append(evidence, fmt.Sprintf("Method: %s", method))
	}

	result := TableClassification{
		TableName:      tableName,
		Classification: classification,
		Confidence:     confidence,
		Evidence:       evidence,
		Source:         sourceID,
		Props:          make(map[string]any),
	}
	result.Props["quality_score"] = qualityScore
	result.Props["needs_review"] = needsReview
	return result, qualityScore, needsReview
}

// classifyTableWithFullSAPRPT calls the Python script to classify using full SAP_RPT_OSS_Classifier with training data
func (ae *AdvancedExtractor) classifyTableWithFullSAPRPT(tableName, context, sourceID, trainingDataPath string) TableClassification {
	// Extract columns from context if possible
	columns := []map[string]any{}
	// This is a simplified extraction - in practice, you'd parse DDL or JSON properly
	if strings.Contains(context, "CREATE TABLE") {
		// Try to extract column definitions from DDL
		columnPattern := regexp.MustCompile(`(?i)(\w+)\s+(\w+)`)
		matches := columnPattern.FindAllStringSubmatch(context, -1)
		for _, match := range matches {
			if len(match) >= 3 {
				columns = append(columns, map[string]any{
					"name": match[1],
					"type": match[2],
				})
			}
		}
	}

	columnsJSON, err := json.Marshal(columns)
	if err != nil {
		return TableClassification{
			TableName:      tableName,
			Classification: "unknown",
			Confidence:     0.0,
			Source:         sourceID,
		}
	}

	cmd := exec.Command("python3", "./scripts/classify_table_sap_rpt_full.py",
		"--table-name", tableName,
		"--columns", string(columnsJSON),
		"--context", context,
		"--training-data", trainingDataPath,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			ae.logger.Printf("full sap-rpt classification failed: %v, stderr: %s", err, string(exitErr.Stderr))
		}
		return TableClassification{
			TableName:      tableName,
			Classification: "unknown",
			Confidence:     0.0,
			Source:         sourceID,
			Props:          make(map[string]any),
		}
	}

	var result map[string]any
	if err := json.Unmarshal(output, &result); err != nil {
		return TableClassification{
			TableName:      tableName,
			Classification: "unknown",
			Confidence:     0.0,
			Source:         sourceID,
			Props:          make(map[string]any),
		}
	}

	classification := "unknown"
	if cls, ok := result["classification"].(string); ok {
		classification = cls
	}

	confidence := 0.0
	if conf, ok := result["confidence"].(float64); ok {
		confidence = conf
	}

	evidence := []string{}
	if ev, ok := result["evidence"].([]interface{}); ok {
		for _, e := range ev {
			if str, ok := e.(string); ok {
				evidence = append(evidence, str)
			}
		}
	}

	return TableClassification{
		TableName:      tableName,
		Classification: classification,
		Confidence:     confidence,
		Evidence:       evidence,
		Source:         sourceID,
		Props:          make(map[string]any),
	}
}

// Helper functions

func parseValueList(valuesStr string) []string {
	values := []string{}
	// Simple parsing - split by comma and trim
	parts := strings.Split(valuesStr, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		// Remove quotes if present
		part = strings.Trim(part, `"'`)
		if part != "" {
			values = append(values, part)
		}
	}
	return values
}

// containsAny returns true if s contains any of the substrings in patterns
func containsAny(s string, patterns []string) bool {
    for _, p := range patterns {
        if strings.Contains(s, p) {
            return true
        }
    }
    return false
}

// classifyTableWithSAPRPT provides a basic fallback if full SAP-RPT classifier is unavailable
func (ae *AdvancedExtractor) classifyTableWithSAPRPT(tableName, context, sourceID string) TableClassification {
    return TableClassification{
        TableName:      tableName,
        Classification: "unknown",
        Confidence:     0.0,
        Evidence:       []string{"sap-rpt fallback used"},
        Patterns:       []string{},
    }
}

