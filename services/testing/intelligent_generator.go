package testing

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"
)

// IntelligentGenerator enhances sample generation with pattern learning and intelligence.
type IntelligentGenerator struct {
	generator *SampleGenerator
	logger    *log.Logger
	patterns  *LearnedPatterns
}

// LearnedPatterns stores patterns learned from historical data.
type LearnedPatterns struct {
	ColumnPatterns     map[string]*ColumnPattern
	ValueDistributions map[string]*ValueDistribution
	RelationshipPatterns map[string]*RelationshipPattern
}

// ColumnPattern represents learned patterns for a column.
type ColumnPattern struct {
	ColumnName    string
	ValuePatterns []string
	EnumValues    []string
	MinValue      any
	MaxValue      any
	CommonValues  []any
	NullRatio     float64
}

// ValueDistribution represents value distribution patterns.
type ValueDistribution struct {
	ColumnName string
	Distribution map[any]int // value -> count
	Cardinality  int
}

// RelationshipPattern represents patterns in table relationships.
type RelationshipPattern struct {
	SourceTable string
	TargetTable string
	Pattern     string
	Frequency   int
}

// NewIntelligentGenerator creates a new intelligent generator.
func NewIntelligentGenerator(generator *SampleGenerator, logger *log.Logger) *IntelligentGenerator {
	return &IntelligentGenerator{
		generator: generator,
		logger:    logger,
		patterns: &LearnedPatterns{
			ColumnPatterns:      make(map[string]*ColumnPattern),
			ValueDistributions:  make(map[string]*ValueDistribution),
			RelationshipPatterns: make(map[string]*RelationshipPattern),
		},
	}
}

// LearnPatternsFromDatabase learns patterns from existing database data.
func (ig *IntelligentGenerator) LearnPatternsFromDatabase(ctx context.Context, db *sql.DB, tableName string) error {
	ig.logger.Printf("Learning patterns from table: %s", tableName)
	
	// Query table to learn patterns
	query := fmt.Sprintf("SELECT * FROM %s LIMIT 10000", tableName)
	rows, err := db.QueryContext(ctx, query)
	if err != nil {
		return fmt.Errorf("query table: %w", err)
	}
	defer rows.Close()
	
	columns, err := rows.Columns()
	if err != nil {
		return fmt.Errorf("get columns: %w", err)
	}
	
	// Initialize column patterns
	columnPatterns := make(map[string]*ColumnPattern)
	valueDistributions := make(map[string]*ValueDistribution)
	
	for _, col := range columns {
		columnPatterns[col] = &ColumnPattern{
			ColumnName:   col,
			ValuePatterns: []string{},
			EnumValues:   []string{},
			CommonValues: []any{},
		}
		valueDistributions[col] = &ValueDistribution{
			ColumnName:  col,
			Distribution: make(map[any]int),
		}
	}
	
	// Scan rows and learn patterns
	rowCount := 0
	nullCounts := make(map[string]int)
	
	for rows.Next() {
		values := make([]any, len(columns))
		valuePtrs := make([]any, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}
		
		if err := rows.Scan(valuePtrs...); err != nil {
			continue
		}
		
		rowCount++
		
		for i, col := range columns {
			val := values[i]
			
			if val == nil {
				nullCounts[col]++
				continue
			}
			
			// Track value distribution
			valueDistributions[col].Distribution[val]++
			
			// Learn patterns based on value type
			ig.learnValuePattern(columnPatterns[col], col, val)
		}
	}
	
	// Calculate patterns
	for _, col := range columns {
		pattern := columnPatterns[col]
		distribution := valueDistributions[col]
		
		// Calculate null ratio
		pattern.NullRatio = float64(nullCounts[col]) / float64(rowCount)
		
		// Find most common values
		maxCount := 0
		for val, count := range distribution.Distribution {
			if count > maxCount {
				maxCount = count
				pattern.CommonValues = []any{val}
			} else if count == maxCount {
				pattern.CommonValues = append(pattern.CommonValues, val)
			}
		}
		
		// Determine if column is enum-like (low cardinality)
		distribution.Cardinality = len(distribution.Distribution)
		if distribution.Cardinality < 20 && distribution.Cardinality < rowCount/10 {
			// Extract enum values
			for val := range distribution.Distribution {
				if str, ok := val.(string); ok {
					pattern.EnumValues = append(pattern.EnumValues, str)
				}
			}
		}
		
		ig.patterns.ColumnPatterns[fmt.Sprintf("%s.%s", tableName, col)] = pattern
		ig.patterns.ValueDistributions[fmt.Sprintf("%s.%s", tableName, col)] = distribution
	}
	
	ig.logger.Printf("Learned patterns for %d columns from %d rows", len(columns), rowCount)
	return nil
}

// learnValuePattern learns patterns from a value.
func (ig *IntelligentGenerator) learnValuePattern(pattern *ColumnPattern, columnName string, value any) {
	colLower := strings.ToLower(columnName)
	
	// Learn email patterns
	if strings.Contains(colLower, "email") {
		if str, ok := value.(string); ok {
			if strings.Contains(str, "@") {
				pattern.ValuePatterns = append(pattern.ValuePatterns, "email")
			}
		}
	}
	
	// Learn date patterns
	if strings.Contains(colLower, "date") || strings.Contains(colLower, "time") {
		if str, ok := value.(string); ok {
			if strings.Contains(str, "-") || strings.Contains(str, "/") {
				pattern.ValuePatterns = append(pattern.ValuePatterns, "date")
			}
		}
	}
	
	// Learn code patterns
	if strings.Contains(colLower, "code") || strings.Contains(colLower, "id") {
		if str, ok := value.(string); ok {
			if len(str) > 0 && (str[0] >= 'A' && str[0] <= 'Z') {
				pattern.ValuePatterns = append(pattern.ValuePatterns, "code_prefix")
			}
		}
	}
}

// GenerateIntelligentValue generates a value using learned patterns.
func (ig *IntelligentGenerator) GenerateIntelligentValue(tableName, columnName string, column *ColumnSchema) any {
	key := fmt.Sprintf("%s.%s", tableName, columnName)
	
	// Use learned patterns if available
	if pattern, exists := ig.patterns.ColumnPatterns[key]; exists {
		// Use enum values if available
		if len(pattern.EnumValues) > 0 {
			return pattern.EnumValues[rand.Intn(len(pattern.EnumValues))]
		}
		
		// Use common values
		if len(pattern.CommonValues) > 0 {
			return pattern.CommonValues[rand.Intn(len(pattern.CommonValues))]
		}
		
		// Use learned patterns
		if len(pattern.ValuePatterns) > 0 {
			patternType := pattern.ValuePatterns[0]
			switch patternType {
			case "email":
				return fmt.Sprintf("user%d@example.com", rand.Intn(1000))
			case "date":
				return time.Now().AddDate(0, 0, -rand.Intn(365)).Format("2006-01-02")
			case "code_prefix":
				return fmt.Sprintf("CODE_%04d", rand.Intn(10000))
			}
		}
	}
	
	// Fallback to standard generation
	return ig.generator.generateValueForColumn(context.Background(), column, &TableSchema{}, &TableTestConfig{}, rand.Intn(1000))
}

