package murex

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/graph/pkg/connectors"
)

// MurexTerminologyExtractor extracts terminology and training data from Murex API.
type MurexTerminologyExtractor struct {
	connector   connectors.Connector
	logger      *log.Logger
	terminology *ExtractedTerminology
	trainingData  *TrainingData
	domain      string // Domain name for terminology extraction
}

// ExtractedTerminology contains terminology extracted from Murex.
type ExtractedTerminology struct {
	Domains        map[string][]TerminologyExample
	Roles          map[string][]TerminologyExample
	NamingPatterns map[string][]TerminologyExample
	FieldDescriptions map[string]string // field_name -> description
	EntityTypes    []string
	Relationships  []string
}

// TerminologyExample represents a single terminology example.
type TerminologyExample struct {
	Text        string
	Context     map[string]interface{}
	Confidence  float64
	Source      string // "openapi_spec", "api_data", "schema"
	Timestamp   time.Time
}

// TrainingData contains training examples extracted from Murex.
type TrainingData struct {
	SchemaExamples   []SchemaExample
	FieldExamples    []FieldExample
	RelationshipExamples []RelationshipExample
	ValuePatterns    []ValuePattern
}

// SchemaExample represents a schema pattern for training.
type SchemaExample struct {
	TableName    string
	Columns      []ColumnInfo
	PrimaryKey   []string
	ForeignKeys  []ForeignKeyInfo
	Description  string
	Source       string
}

// ColumnInfo represents column information.
type ColumnInfo struct {
	Name        string
	Type        string
	Description string
	Nullable    bool
	Examples    []interface{} // Sample values from actual data
}

// ForeignKeyInfo represents foreign key information.
type ForeignKeyInfo struct {
	Columns          []string
	ReferencedTable  string
	ReferencedColumns []string
}

// FieldExample represents a field pattern for training.
type FieldExample struct {
	FieldName    string
	FieldType    string
	Domain       string
	Role         string
	Pattern      string
	Examples     []interface{}
	Description  string
}

// RelationshipExample represents a relationship pattern.
type RelationshipExample struct {
	SourceType   string
	TargetType   string
	Relationship string
	Description  string
}

// ValuePattern represents a value pattern extracted from data.
type ValuePattern struct {
	FieldName    string
	Pattern      string // regex or pattern description
	Examples     []interface{}
	Frequency    int
}

func NewMurexTerminologyExtractor(conn connectors.Connector, logger *log.Logger) *MurexTerminologyExtractor {
	return &MurexTerminologyExtractor{
		connector: conn,
		logger: logger,
		terminology: &ExtractedTerminology{
			Domains:        make(map[string][]TerminologyExample),
			Roles:          make(map[string][]TerminologyExample),
			NamingPatterns: make(map[string][]TerminologyExample),
			FieldDescriptions: make(map[string]string),
			EntityTypes:    []string{},
			Relationships:  []string{},
		},
		trainingData: &TrainingData{
			SchemaExamples:   []SchemaExample{},
			FieldExamples:    []FieldExample{},
			RelationshipExamples: []RelationshipExample{},
			ValuePatterns:    []ValuePattern{},
		},
		domain: "finance-risk-treasury", // Default domain
	}
}

// ExtractFromOpenAPISpec extracts terminology from the OpenAPI specification.
func (mte *MurexTerminologyExtractor) ExtractFromOpenAPISpec(ctx context.Context) error {
	if mte.logger != nil {
		mte.logger.Printf("Extracting terminology from Murex OpenAPI specification")
	}

	// Discover schema from OpenAPI
	schema, err := mte.connector.DiscoverSchema(ctx)
	if err != nil {
		return fmt.Errorf("failed to discover schema: %w", err)
	}

	if schema == nil {
		return fmt.Errorf("schema is nil")
	}

	// Extract domain terminology from table names
	mte.extractDomainTerminology(schema)

	// Extract field terminology from column definitions
	mte.extractFieldTerminology(schema)

	// Extract relationship terminology
	mte.extractRelationshipTerminology(schema)

	// Extract naming patterns
	mte.extractNamingPatterns(schema)

	if mte.logger != nil {
		mte.logger.Printf("Extracted terminology: %d domains, %d roles, %d patterns",
			len(mte.terminology.Domains),
			len(mte.terminology.Roles),
			len(mte.terminology.NamingPatterns))
	}

	return nil
}

// ExtractFromAPIData extracts terminology and training patterns from actual API data.
func (mte *MurexTerminologyExtractor) ExtractFromAPIData(ctx context.Context, sampleSize int) error {
	if mte.logger != nil {
		mte.logger.Printf("Extracting terminology from Murex API data (sample_size=%d)", sampleSize)
	}

	// Connect to Murex (only if not already connected)
	// Note: Connector should handle multiple Connect calls gracefully
	if err := mte.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	
	// Only close if we opened the connection
	// In production, connector should manage its own lifecycle
	defer func() {
		if closeErr := mte.connector.Close(); closeErr != nil && mte.logger != nil {
			mte.logger.Printf("Warning: Error closing connector: %v", closeErr)
		}
	}()

	// Discover schema to know what tables to sample
	schema, err := mte.connector.DiscoverSchema(ctx)
	if err != nil {
		return fmt.Errorf("failed to discover schema: %w", err)
	}

	if schema == nil {
		return fmt.Errorf("schema is nil")
	}

	// Sample data from each table
	for _, table := range schema.Tables {
		query := map[string]interface{}{
			"table": table.Name,
			"limit": sampleSize,
		}

		data, err := mte.connector.ExtractData(ctx, query)
		if err != nil {
			if mte.logger != nil {
				mte.logger.Printf("Warning: Failed to extract data from %s: %v", table.Name, err)
			}
			continue
		}

		// Extract patterns from actual data
		mte.extractDataPatterns(table, data)

		// Extract value patterns
		mte.extractValuePatterns(table, data)

		// Extract field examples
		mte.extractFieldExamplesFromData(table, data)
	}

	if mte.logger != nil {
		mte.logger.Printf("Extracted training data: %d schema examples, %d field examples, %d value patterns",
			len(mte.trainingData.SchemaExamples),
			len(mte.trainingData.FieldExamples),
			len(mte.trainingData.ValuePatterns))
	}

	return nil
}

// extractDomainTerminology extracts domain terminology from schema.
func (mte *MurexTerminologyExtractor) extractDomainTerminology(schema *SourceSchema) {
	// Finance-Risk-Treasury domain terms
	financeDomain := []TerminologyExample{
		{Text: "trade", Context: map[string]interface{}{"source": "murex"}, Confidence: 0.95, Source: "openapi_spec", Timestamp: time.Now()},
		{Text: "cashflow", Context: map[string]interface{}{"source": "murex"}, Confidence: 0.95, Source: "openapi_spec", Timestamp: time.Now()},
		{Text: "position", Context: map[string]interface{}{"source": "murex"}, Confidence: 0.90, Source: "openapi_spec", Timestamp: time.Now()},
		{Text: "counterparty", Context: map[string]interface{}{"source": "murex"}, Confidence: 0.95, Source: "openapi_spec", Timestamp: time.Now()},
		{Text: "instrument", Context: map[string]interface{}{"source": "murex"}, Confidence: 0.90, Source: "openapi_spec", Timestamp: time.Now()},
		{Text: "pricing", Context: map[string]interface{}{"source": "murex"}, Confidence: 0.85, Source: "openapi_spec", Timestamp: time.Now()},
		{Text: "market_data", Context: map[string]interface{}{"source": "murex"}, Confidence: 0.85, Source: "openapi_spec", Timestamp: time.Now()},
	}

	mte.terminology.Domains[mte.domain] = financeDomain
	mte.terminology.EntityTypes = append(mte.terminology.EntityTypes, "Trade", "Cashflow", "Position", "Counterparty", "Instrument")

	// Add table-specific domain terms
	for _, table := range schema.Tables {
		term := TerminologyExample{
			Text:        table.Name,
			Context:     map[string]interface{}{"table": table.Name, "source": "murex"},
			Confidence:  0.90,
			Source:      "openapi_spec",
			Timestamp:   time.Now(),
		}
		mte.terminology.Domains[mte.domain] = append(mte.terminology.Domains[mte.domain], term)
	}
}

// extractFieldTerminology extracts field/role terminology.
func (mte *MurexTerminologyExtractor) extractFieldTerminology(schema *SourceSchema) {
	roleMappings := map[string]string{
		"trade_id":      "identifier",
		"id":            "identifier",
		"notional":      "amount",
		"amount":        "amount",
		"price":         "amount",
		"value":         "amount",
		"trade_date":    "date",
		"date":          "date",
		"timestamp":     "date",
		"created_at":    "date",
		"currency":      "code",
		"status":        "status",
		"counterparty":  "reference",
		"instrument":    "reference",
		"description":   "text",
		"name":          "name",
	}

	for _, table := range schema.Tables {
		for _, column := range table.Columns {
			// Infer role from column name
			role := mte.inferRole(column.Name, column.Type)
			
			term := TerminologyExample{
				Text:        column.Name,
				Context:     map[string]interface{}{"table": table.Name, "type": column.Type, "role": role},
				Confidence:  0.85,
				Source:      "openapi_spec",
				Timestamp:   time.Now(),
			}

			if roleMappings[strings.ToLower(column.Name)] != "" {
				role = roleMappings[strings.ToLower(column.Name)]
			}

			mte.terminology.Roles[role] = append(mte.terminology.Roles[role], term)
		}
	}
}

// inferRole infers the business role from field name and type.
func (mte *MurexTerminologyExtractor) inferRole(fieldName, fieldType string) string {
	lowerName := strings.ToLower(fieldName)

	// Pattern matching for common roles
	if strings.Contains(lowerName, "id") || strings.HasSuffix(lowerName, "_id") {
		return "identifier"
	}
	if strings.Contains(lowerName, "amount") || strings.Contains(lowerName, "notional") || 
	   strings.Contains(lowerName, "price") || strings.Contains(lowerName, "value") {
		return "amount"
	}
	if strings.Contains(lowerName, "date") || strings.Contains(lowerName, "time") {
		return "date"
	}
	if strings.Contains(lowerName, "currency") || strings.Contains(lowerName, "ccy") {
		return "code"
	}
	if strings.Contains(lowerName, "status") || strings.Contains(lowerName, "state") {
		return "status"
	}
	if strings.Contains(lowerName, "counterparty") || strings.Contains(lowerName, "party") {
		return "reference"
	}
	if strings.Contains(lowerName, "description") || strings.Contains(lowerName, "comment") {
		return "text"
	}
	if strings.Contains(lowerName, "name") {
		return "name"
	}

	return "unknown"
}

// extractNamingPatterns extracts naming convention patterns.
func (mte *MurexTerminologyExtractor) extractNamingPatterns(schema *SourceSchema) {
	patterns := map[string]int{
		"snake_case":     0,
		"camelCase":      0,
		"PascalCase":     0,
		"UPPER_SNAKE":    0,
		"has_id_suffix":  0,
		"has_id_prefix":  0,
	}

	for _, table := range schema.Tables {
		// Analyze table name
		if strings.Contains(table.Name, "_") {
			patterns["snake_case"]++
		}
		if strings.HasSuffix(table.Name, "_id") {
			patterns["has_id_suffix"]++
		}

		// Analyze column names
		for _, column := range table.Columns {
			colName := column.Name
			if strings.Contains(colName, "_") {
				patterns["snake_case"]++
			}
			if strings.HasSuffix(colName, "_id") {
				patterns["has_id_suffix"]++
			}
			if strings.HasPrefix(colName, "id") {
				patterns["has_id_prefix"]++
			}
		}
	}

	// Create pattern examples
	for pattern, count := range patterns {
		if count > 0 {
			term := TerminologyExample{
				Text:        pattern,
				Context:     map[string]interface{}{"frequency": count, "source": "murex"},
				Confidence:  0.80,
				Source:      "openapi_spec",
				Timestamp:   time.Now(),
			}
			mte.terminology.NamingPatterns[pattern] = append(mte.terminology.NamingPatterns[pattern], term)
		}
	}
}

// extractRelationshipTerminology extracts relationship terminology.
func (mte *MurexTerminologyExtractor) extractRelationshipTerminology(schema *SourceSchema) {
	// Note: SourceTable doesn't have ForeignKeys in the simplified schema
	// This is a stub implementation - in production, foreign keys would be discovered separately
	for _, table := range schema.Tables {
		// Extract relationships from table names and metadata if available
		if metadata, ok := schema.Metadata["foreign_keys"].(map[string]interface{}); ok {
			for _, fkInfo := range metadata {
				if fkMap, ok := fkInfo.(map[string]interface{}); ok {
					if refTable, ok := fkMap["referenced_table"].(string); ok {
						rel := fmt.Sprintf("%s -> %s", table.Name, refTable)
						mte.terminology.Relationships = append(mte.terminology.Relationships, rel)

						example := RelationshipExample{
							SourceType:   table.Name,
							TargetType:   refTable,
							Relationship: "references",
							Description:  fmt.Sprintf("%s references %s", table.Name, refTable),
						}
						mte.trainingData.RelationshipExamples = append(mte.trainingData.RelationshipExamples, example)
					}
				}
			}
		}
	}
}

// extractDataPatterns extracts patterns from actual data.
func (mte *MurexTerminologyExtractor) extractDataPatterns(table SourceTable, data []map[string]interface{}) {
	if len(data) == 0 {
		return
	}

	// Create schema example
	primaryKey := []string{}
	if table.PrimaryKey != "" {
		primaryKey = []string{table.PrimaryKey}
	}

	schemaExample := SchemaExample{
		TableName:   table.Name,
		Columns:     []ColumnInfo{},
		PrimaryKey:  primaryKey,
		Description: fmt.Sprintf("Murex %s table", table.Name),
		Source:      "murex_api",
	}

	// Extract column info with example values
	for _, column := range table.Columns {
		colInfo := ColumnInfo{
			Name:        column.Name,
			Type:        column.Type,
			Nullable:    false, // Default to false since SourceColumn doesn't have this field
			Examples:    []interface{}{},
		}

		// Collect sample values from data
		for _, record := range data {
			if val, ok := record[column.Name]; ok && val != nil {
				colInfo.Examples = append(colInfo.Examples, val)
				if len(colInfo.Examples) >= 5 { // Limit examples
					break
				}
			}
		}

		schemaExample.Columns = append(schemaExample.Columns, colInfo)
	}

	// Note: ForeignKeys not available in simplified SourceTable - would need to be discovered separately
	mte.trainingData.SchemaExamples = append(mte.trainingData.SchemaExamples, schemaExample)
}

// extractValuePatterns extracts value patterns from data.
func (mte *MurexTerminologyExtractor) extractValuePatterns(table SourceTable, data []map[string]interface{}) {
	if len(data) == 0 {
		return
	}

	// Analyze patterns for each column
	for _, column := range table.Columns {
		valuePattern := ValuePattern{
			FieldName: column.Name,
			Examples:  []interface{}{},
			Frequency: 0,
		}

		// Collect values and infer pattern
		for _, record := range data {
			if val, ok := record[column.Name]; ok && val != nil {
				valuePattern.Examples = append(valuePattern.Examples, val)
				valuePattern.Frequency++
				if len(valuePattern.Examples) >= 10 {
					break
				}
			}
		}

		// Infer pattern type
		if len(valuePattern.Examples) > 0 {
			valuePattern.Pattern = mte.inferValuePattern(column.Type, valuePattern.Examples)
			mte.trainingData.ValuePatterns = append(mte.trainingData.ValuePatterns, valuePattern)
		}
	}
}

// inferValuePattern infers the pattern type from examples.
func (mte *MurexTerminologyExtractor) inferValuePattern(fieldType string, examples []interface{}) string {
	if len(examples) == 0 {
		return "unknown"
	}

	// More robust pattern inference
	switch fieldType {
	case "string":
		// Analyze multiple examples if available
		sampleSize := len(examples)
		if sampleSize > 10 {
			sampleSize = 10 // Limit sample size
		}
		
		patterns := make(map[string]int)
		for i := 0; i < sampleSize; i++ {
			val := fmt.Sprintf("%v", examples[i])
			valLen := len(val)
			valUpper := strings.ToUpper(val)
			
			// Currency code pattern (3 uppercase letters)
			if valLen == 3 && val == valUpper {
				patterns["currency_code"]++
			}
			// Date string pattern (YYYY-MM-DD or similar)
			if strings.Contains(val, "-") && (valLen == 10 || valLen == 19) {
				patterns["date_string"]++
			}
			// ISO 8601 datetime pattern
			if strings.Contains(val, "T") && strings.Contains(val, "Z") {
				patterns["iso_datetime"]++
			}
			// UUID pattern
			if valLen == 36 && strings.Contains(val, "-") {
				patterns["uuid"]++
			}
			// Numeric string
			if _, err := strconv.ParseFloat(val, 64); err == nil {
				patterns["numeric_string"]++
			}
		}
		
		// Return most common pattern
		if len(patterns) > 0 {
			maxCount := 0
			commonPattern := "text"
			for pattern, count := range patterns {
				if count > maxCount {
					maxCount = count
					commonPattern = pattern
				}
			}
			// Only return specialized pattern if it appears in majority
			if maxCount >= sampleSize/2 {
				return commonPattern
			}
		}
		return "text"
	case "decimal", "float", "double", "numeric":
		// Check if values are monetary (typically 2 decimal places)
		if len(examples) > 0 {
			valStr := fmt.Sprintf("%v", examples[0])
			if strings.Contains(valStr, ".") {
				parts := strings.Split(valStr, ".")
				if len(parts) == 2 && len(parts[1]) == 2 {
					return "monetary"
				}
			}
		}
		return "numeric"
	case "integer", "int", "bigint":
		return "integer"
	case "date", "datetime", "timestamp":
		return "date"
	case "boolean", "bool":
		return "boolean"
	default:
		return "unknown"
	}
}

// extractFieldExamplesFromData extracts field examples from actual data.
func (mte *MurexTerminologyExtractor) extractFieldExamplesFromData(table SourceTable, data []map[string]interface{}) {
	for _, column := range table.Columns {
		domain := mte.domain // Use configured domain
		role := mte.inferRole(column.Name, column.Type)

		examples := []interface{}{}
		for _, record := range data {
			if val, ok := record[column.Name]; ok && val != nil {
				examples = append(examples, val)
				if len(examples) >= 5 {
					break
				}
			}
		}

		fieldExample := FieldExample{
			FieldName:   column.Name,
			FieldType:   column.Type,
			Domain:      domain,
			Role:        role,
			Pattern:     mte.inferNamingPattern(column.Name),
			Examples:    examples,
			Description: fmt.Sprintf("Murex %s.%s", table.Name, column.Name),
		}

		mte.trainingData.FieldExamples = append(mte.trainingData.FieldExamples, fieldExample)
	}
}

// inferNamingPattern infers naming pattern from field name.
func (mte *MurexTerminologyExtractor) inferNamingPattern(fieldName string) string {
	if strings.Contains(fieldName, "_") {
		return "snake_case"
	}
	if len(fieldName) > 0 && fieldName[0] >= 'A' && fieldName[0] <= 'Z' {
		return "PascalCase"
	}
	if strings.ToUpper(fieldName) == fieldName {
		return "UPPER_SNAKE"
	}
	return "camelCase"
}

// GetTerminology returns the extracted terminology.
func (mte *MurexTerminologyExtractor) GetTerminology() *ExtractedTerminology {
	return mte.terminology
}

// GetTrainingData returns the extracted training data.
func (mte *MurexTerminologyExtractor) GetTrainingData() *TrainingData {
	return mte.trainingData
}

