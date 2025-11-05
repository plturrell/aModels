package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
)

// CrossSystemExtractor extracts patterns across multiple systems and platforms.
type CrossSystemExtractor struct {
	logger            *log.Logger
	terminologyLearner *TerminologyLearner // Phase 10: LNN-based terminology learning
}

// NewCrossSystemExtractor creates a new cross-system extractor.
func NewCrossSystemExtractor(logger *log.Logger) *CrossSystemExtractor {
	return &CrossSystemExtractor{
		logger:            logger,
		terminologyLearner: nil, // Will be set via SetTerminologyLearner
	}
}

// SetTerminologyLearner sets the terminology learner (Phase 10).
func (cse *CrossSystemExtractor) SetTerminologyLearner(learner *TerminologyLearner) {
	cse.terminologyLearner = learner
}

// SystemSchema represents a schema from a specific system.
type SystemSchema struct {
	SystemType   string            `json:"system_type"`   // "postgres", "hana", "mysql", etc.
	SystemName   string            `json:"system_name"`
	DatabaseName string            `json:"database_name"`
	SchemaName   string            `json:"schema_name"`
	Tables       []TableSchema     `json:"tables"`
	Metadata     map[string]any    `json:"metadata,omitempty"`
}

// TableSchema represents a table schema.
type TableSchema struct {
	TableName  string            `json:"table_name"`
	Columns    []ColumnSchema    `json:"columns"`
	PrimaryKey []string          `json:"primary_key,omitempty"`
	ForeignKeys []ForeignKeySchema `json:"foreign_keys,omitempty"`
	Indexes    []string          `json:"indexes,omitempty"`
	Metadata   map[string]any    `json:"metadata,omitempty"`
}

// ColumnSchema represents a column schema.
type ColumnSchema struct {
	ColumnName string `json:"column_name"`
	DataType   string `json:"data_type"`
	Nullable   bool   `json:"nullable"`
	Default    string `json:"default,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// ForeignKeySchema represents a foreign key relationship.
type ForeignKeySchema struct {
	Name            string `json:"name"`
	Column          string `json:"column"`
	ReferencedTable string `json:"referenced_table"`
	ReferencedColumn string `json:"referenced_column"`
}

// CrossSystemPattern represents a pattern found across multiple systems.
type CrossSystemPattern struct {
	PatternID       string            `json:"pattern_id"`
	PatternType     string            `json:"pattern_type"`     // "table_structure", "naming_convention", "relationship", etc.
	PatternName     string            `json:"pattern_name"`
	AffectedSystems []string          `json:"affected_systems"`
	Occurrences     []PatternOccurrence `json:"occurrences"`
	Confidence      float64           `json:"confidence"`       // 0.0 to 1.0
	NormalizedForm  map[string]any    `json:"normalized_form"` // Universal pattern template
}

// PatternOccurrence represents an occurrence of a pattern in a specific system.
type PatternOccurrence struct {
	SystemType   string         `json:"system_type"`
	SystemName   string         `json:"system_name"`
	DatabaseName string         `json:"database_name"`
	TableName    string         `json:"table_name"`
	Details      map[string]any `json:"details"`
}

// ExtractCrossSystemPatterns extracts patterns across multiple systems.
func (cse *CrossSystemExtractor) ExtractCrossSystemPatterns(
	ctx context.Context,
	schemas []SystemSchema,
) ([]CrossSystemPattern, error) {
	patterns := []CrossSystemPattern{}

	// Extract table structure patterns
	tablePatterns, err := cse.extractTableStructurePatterns(schemas)
	if err != nil {
		cse.logger.Printf("failed to extract table structure patterns: %v", err)
	} else {
		patterns = append(patterns, tablePatterns...)
	}

	// Extract naming convention patterns
	namingPatterns, err := cse.extractNamingConventionPatterns(schemas)
	if err != nil {
		cse.logger.Printf("failed to extract naming convention patterns: %v", err)
	} else {
		patterns = append(patterns, namingPatterns...)
	}

	// Extract relationship patterns
	relationshipPatterns, err := cse.extractRelationshipPatterns(schemas)
	if err != nil {
		cse.logger.Printf("failed to extract relationship patterns: %v", err)
	} else {
		patterns = append(patterns, relationshipPatterns...)
	}

	// Normalize patterns to universal templates
	cse.normalizePatterns(patterns)

	return patterns, nil
}

// extractTableStructurePatterns extracts patterns in table structures.
func (cse *CrossSystemExtractor) extractTableStructurePatterns(
	schemas []SystemSchema,
) ([]CrossSystemPattern, error) {
	patterns := []CrossSystemPattern{}

	// Group tables by structure similarity
	structureGroups := make(map[string][]PatternOccurrence)

	for _, schema := range schemas {
		for _, table := range schema.Tables {
			// Create a normalized structure signature
			signature := cse.createStructureSignature(table)

			occurrence := PatternOccurrence{
				SystemType:   schema.SystemType,
				SystemName:   schema.SystemName,
				DatabaseName: schema.DatabaseName,
				TableName:    table.TableName,
				Details: map[string]any{
					"column_count": len(table.Columns),
					"primary_key_count": len(table.PrimaryKey),
					"foreign_key_count": len(table.ForeignKeys),
				},
			}

			structureGroups[signature] = append(structureGroups[signature], occurrence)
		}
	}

	// Create patterns for structures that appear in multiple systems
	patternCounter := 0
	for signature, occurrences := range structureGroups {
		if len(occurrences) > 1 {
			// Pattern appears in multiple systems
			systems := make(map[string]bool)
			for _, occ := range occurrences {
				systems[occ.SystemType] = true
			}

			if len(systems) > 1 {
				// Cross-system pattern found
				patternID := fmt.Sprintf("table_structure_%d", patternCounter)
				patternCounter++

				systemList := []string{}
				for sys := range systems {
					systemList = append(systemList, sys)
				}

				pattern := CrossSystemPattern{
					PatternID:       patternID,
					PatternType:     "table_structure",
					PatternName:     fmt.Sprintf("Table Structure Pattern: %s", signature),
					AffectedSystems: systemList,
					Occurrences:     occurrences,
					Confidence:      cse.calculatePatternConfidence(occurrences),
				}

				patterns = append(patterns, pattern)
			}
		}
	}

	return patterns, nil
}

// extractNamingConventionPatterns extracts naming convention patterns.
func (cse *CrossSystemExtractor) extractNamingConventionPatterns(
	schemas []SystemSchema,
) ([]CrossSystemPattern, error) {
	patterns := []CrossSystemPattern{}

	// Group columns by naming patterns
	namingGroups := make(map[string][]PatternOccurrence)

	for _, schema := range schemas {
		for _, table := range schema.Tables {
			for _, column := range table.Columns {
				// Analyze naming convention
				namingPattern := cse.analyzeNamingConvention(column.ColumnName)

				occurrence := PatternOccurrence{
					SystemType:   schema.SystemType,
					SystemName:   schema.SystemName,
					DatabaseName: schema.DatabaseName,
					TableName:    table.TableName,
					Details: map[string]any{
						"column_name": column.ColumnName,
						"naming_pattern": namingPattern,
					},
				}

				namingGroups[namingPattern] = append(namingGroups[namingPattern], occurrence)
			}
		}
	}

	// Create patterns for naming conventions that appear across systems
	patternCounter := 0
	for namingPattern, occurrences := range namingGroups {
		if len(occurrences) > 1 {
			systems := make(map[string]bool)
			for _, occ := range occurrences {
				systems[occ.SystemType] = true
			}

			if len(systems) > 1 {
				patternID := fmt.Sprintf("naming_convention_%d", patternCounter)
				patternCounter++

				systemList := []string{}
				for sys := range systems {
					systemList = append(systemList, sys)
				}

				pattern := CrossSystemPattern{
					PatternID:       patternID,
					PatternType:     "naming_convention",
					PatternName:     fmt.Sprintf("Naming Convention: %s", namingPattern),
					AffectedSystems: systemList,
					Occurrences:     occurrences,
					Confidence:      cse.calculatePatternConfidence(occurrences),
				}

				patterns = append(patterns, pattern)
			}
		}
	}

	return patterns, nil
}

// extractRelationshipPatterns extracts relationship patterns.
func (cse *CrossSystemExtractor) extractRelationshipPatterns(
	schemas []SystemSchema,
) ([]CrossSystemPattern, error) {
	patterns := []CrossSystemPattern{}

	// Group foreign keys by relationship pattern
	relationshipGroups := make(map[string][]PatternOccurrence)

	for _, schema := range schemas {
		for _, table := range schema.Tables {
			for _, fk := range table.ForeignKeys {
				// Create relationship signature
				signature := fmt.Sprintf("%s->%s", fk.ReferencedTable, table.TableName)

				occurrence := PatternOccurrence{
					SystemType:   schema.SystemType,
					SystemName:   schema.SystemName,
					DatabaseName: schema.DatabaseName,
					TableName:    table.TableName,
					Details: map[string]any{
						"foreign_key": fk.Name,
						"referenced_table": fk.ReferencedTable,
						"relationship_type": "foreign_key",
					},
				}

				relationshipGroups[signature] = append(relationshipGroups[signature], occurrence)
			}
		}
	}

	// Create patterns for relationships that appear across systems
	patternCounter := 0
	for signature, occurrences := range relationshipGroups {
		if len(occurrences) > 1 {
			systems := make(map[string]bool)
			for _, occ := range occurrences {
				systems[occ.SystemType] = true
			}

			if len(systems) > 1 {
				patternID := fmt.Sprintf("relationship_%d", patternCounter)
				patternCounter++

				systemList := []string{}
				for sys := range systems {
					systemList = append(systemList, sys)
				}

				pattern := CrossSystemPattern{
					PatternID:       patternID,
					PatternType:     "relationship",
					PatternName:     fmt.Sprintf("Relationship Pattern: %s", signature),
					AffectedSystems: systemList,
					Occurrences:     occurrences,
					Confidence:      cse.calculatePatternConfidence(occurrences),
				}

				patterns = append(patterns, pattern)
			}
		}
	}

	return patterns, nil
}

// createStructureSignature creates a normalized signature for a table structure.
func (cse *CrossSystemExtractor) createStructureSignature(table TableSchema) string {
	// Create a signature based on:
	// - Column count
	// - Primary key count
	// - Foreign key count
	// - Column type distribution

	typeCounts := make(map[string]int)
	for _, col := range table.Columns {
		typeCounts[col.DataType]++
	}

	signature := fmt.Sprintf("cols:%d_pk:%d_fk:%d_types:%v",
		len(table.Columns),
		len(table.PrimaryKey),
		len(table.ForeignKeys),
		typeCounts,
	)

	return signature
}

// analyzeNamingConvention analyzes the naming convention of a column name.
// Phase 10: Enhanced with LNN-based pattern recognition
func (cse *CrossSystemExtractor) analyzeNamingConvention(columnName string) string {
	// Phase 10: Use LNN for naming convention detection if available
	if cse.terminologyLearner != nil {
		ctx := context.Background()
		patterns := cse.terminologyLearner.AnalyzeNamingConvention(ctx, columnName)
		if len(patterns) > 0 {
			return patterns[0] // Return first detected pattern
		}
	}

	// Fallback to fixed pattern matching
	lower := strings.ToLower(columnName)

	if strings.Contains(columnName, "_") {
		return "snake_case"
	} else if strings.Contains(columnName, "-") {
		return "kebab-case"
	} else if len(columnName) > 0 && columnName[0] >= 'A' && columnName[0] <= 'Z' {
		return "PascalCase"
	} else if strings.ToLower(columnName) != columnName {
		return "camelCase"
	} else if strings.ToUpper(columnName) == columnName {
		return "UPPER_CASE"
	}

	return "unknown"
}

// calculatePatternConfidence calculates confidence score for a pattern.
func (cse *CrossSystemExtractor) calculatePatternConfidence(occurrences []PatternOccurrence) float64 {
	// Higher confidence if:
	// - More occurrences
	// - More diverse systems
	// - More consistent details

	systemCount := make(map[string]int)
	for _, occ := range occurrences {
		systemCount[occ.SystemType]++
	}

	// Base confidence on number of systems and occurrences
	systemDiversity := float64(len(systemCount)) / float64(len(occurrences))
	occurrenceCount := float64(len(occurrences))

	confidence := min(1.0, (systemDiversity*0.5 + min(1.0, occurrenceCount/10.0)*0.5))

	return confidence
}

// normalizePatterns normalizes patterns to universal templates.
func (cse *CrossSystemExtractor) normalizePatterns(patterns []CrossSystemPattern) {
	for i := range patterns {
		pattern := &patterns[i]

		// Create normalized form based on pattern type
		switch pattern.PatternType {
		case "table_structure":
			pattern.NormalizedForm = cse.normalizeTableStructurePattern(pattern)
		case "naming_convention":
			pattern.NormalizedForm = cse.normalizeNamingConventionPattern(pattern)
		case "relationship":
			pattern.NormalizedForm = cse.normalizeRelationshipPattern(pattern)
		}
	}
}

// normalizeTableStructurePattern normalizes a table structure pattern.
func (cse *CrossSystemExtractor) normalizeTableStructurePattern(pattern *CrossSystemPattern) map[string]any {
	if len(pattern.Occurrences) == 0 {
		return map[string]any{}
	}

	// Get average structure from occurrences
	firstOcc := pattern.Occurrences[0]
	avgColumnCount := 0
	avgPKCount := 0
	avgFKCount := 0

	for _, occ := range pattern.Occurrences {
		if colCount, ok := occ.Details["column_count"].(int); ok {
			avgColumnCount += colCount
		}
		if pkCount, ok := occ.Details["primary_key_count"].(int); ok {
			avgPKCount += pkCount
		}
		if fkCount, ok := occ.Details["foreign_key_count"].(int); ok {
			avgFKCount += fkCount
		}
	}

	count := len(pattern.Occurrences)
	if count > 0 {
		avgColumnCount /= count
		avgPKCount /= count
		avgFKCount /= count
	}

	return map[string]any{
		"avg_column_count": avgColumnCount,
		"avg_primary_key_count": avgPKCount,
		"avg_foreign_key_count": avgFKCount,
		"system_agnostic": true,
	}
}

// normalizeNamingConventionPattern normalizes a naming convention pattern.
func (cse *CrossSystemExtractor) normalizeNamingConventionPattern(pattern *CrossSystemPattern) map[string]any {
	if len(pattern.Occurrences) == 0 {
		return map[string]any{}
	}

	// Extract naming pattern from first occurrence
	firstOcc := pattern.Occurrences[0]
	namingPattern := "unknown"
	if pattern, ok := firstOcc.Details["naming_pattern"].(string); ok {
		namingPattern = pattern
	}

	return map[string]any{
		"naming_convention": namingPattern,
		"system_agnostic": true,
	}
}

// normalizeRelationshipPattern normalizes a relationship pattern.
func (cse *CrossSystemExtractor) normalizeRelationshipPattern(pattern *CrossSystemPattern) map[string]any {
	if len(pattern.Occurrences) == 0 {
		return map[string]any{}
	}

	// Extract relationship type
	relationshipType := "foreign_key"
	if len(pattern.Occurrences) > 0 {
		if relType, ok := pattern.Occurrences[0].Details["relationship_type"].(string); ok {
			relationshipType = relType
		}
	}

	return map[string]any{
		"relationship_type": relationshipType,
		"system_agnostic": true,
	}
}

// CompareSchemas compares schemas from different systems.
func (cse *CrossSystemExtractor) CompareSchemas(
	ctx context.Context,
	schema1 SystemSchema,
	schema2 SystemSchema,
) (map[string]any, error) {
	comparison := map[string]any{
		"system1": schema1.SystemType,
		"system2": schema2.SystemType,
		"similarities": []string{},
		"differences": []string{},
	}

	// Compare table structures
	table1Map := make(map[string]TableSchema)
	for _, table := range schema1.Tables {
		table1Map[table.TableName] = table
	}

	table2Map := make(map[string]TableSchema)
	for _, table := range schema2.Tables {
		table2Map[table.TableName] = table
	}

	commonTables := []string{}
	onlyInSystem1 := []string{}
	onlyInSystem2 := []string{}

	for tableName := range table1Map {
		if _, exists := table2Map[tableName]; exists {
			commonTables = append(commonTables, tableName)
		} else {
			onlyInSystem1 = append(onlyInSystem1, tableName)
		}
	}

	for tableName := range table2Map {
		if _, exists := table1Map[tableName]; !exists {
			onlyInSystem2 = append(onlyInSystem2, tableName)
		}
	}

	comparison["common_tables"] = commonTables
	comparison["only_in_system1"] = onlyInSystem1
	comparison["only_in_system2"] = onlyInSystem2

	// Compare structures of common tables
	structureSimilarities := []string{}
	for _, tableName := range commonTables {
		table1 := table1Map[tableName]
		table2 := table2Map[tableName]

		if len(table1.Columns) == len(table2.Columns) {
			structureSimilarities = append(structureSimilarities,
				fmt.Sprintf("Table %s has same column count", tableName))
		}
	}

	comparison["structure_similarities"] = structureSimilarities

	return comparison, nil
}

// Helper function
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

