package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

// SemanticSchemaAnalyzer performs deep semantic understanding of schemas and data lineage.
// Phase 8.1: Enhanced with domain manager integration for domain-aware semantic analysis.
type SemanticSchemaAnalyzer struct {
	logger              *log.Logger
	useSAPRPTEmbeddings bool
	useGloveEmbeddings  bool
	extractServiceURL   string
	domainDetector      *DomainDetector     // Phase 8.1: Domain detector for domain-aware analysis
	terminologyLearner  *TerminologyLearner // Phase 10: LNN-based terminology learning
}

// NewSemanticSchemaAnalyzer creates a new semantic schema analyzer.
func NewSemanticSchemaAnalyzer(logger *log.Logger) *SemanticSchemaAnalyzer {
	localaiURL := os.Getenv("LOCALAI_URL")
	var domainDetector *DomainDetector
	if localaiURL != "" {
		domainDetector = NewDomainDetector(localaiURL, logger)
	}

	return &SemanticSchemaAnalyzer{
		logger:              logger,
		useSAPRPTEmbeddings: os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true",
		useGloveEmbeddings:  os.Getenv("USE_GLOVE_EMBEDDINGS") == "true",
		extractServiceURL:   os.Getenv("EXTRACT_SERVICE_URL"),
		domainDetector:      domainDetector, // Phase 8.1: Domain detector
		terminologyLearner:  nil,            // Will be set via SetTerminologyLearner (Phase 10)
	}
}

// SetTerminologyLearner sets the terminology learner (Phase 10).
func (ssa *SemanticSchemaAnalyzer) SetTerminologyLearner(learner *TerminologyLearner) {
	ssa.terminologyLearner = learner
}

// SemanticColumnAnalysis contains semantic analysis results for a column.
type SemanticColumnAnalysis struct {
	ColumnName             string             `json:"column_name"`
	TableName              string             `json:"table_name"`
	InferredDomain         string             `json:"inferred_domain"`          // e.g., "financial", "customer", "product"
	InferredBusinessRole   string             `json:"inferred_business_role"`   // e.g., "identifier", "amount", "date"
	SemanticSimilarity     map[string]float64 `json:"semantic_similarity"`      // Similarity to known patterns
	NamingConventions      []string           `json:"naming_conventions"`       // Detected naming patterns
	DomainConfidence       float64            `json:"domain_confidence"`        // 0.0 to 1.0
	BusinessRoleConfidence float64            `json:"business_role_confidence"` // 0.0 to 1.0
	ContextualFeatures     map[string]any     `json:"contextual_features"`      // Context-aware features
}

// SemanticLineageAnalysis contains semantic lineage information.
type SemanticLineageAnalysis struct {
	SourceTable           string   `json:"source_table"`
	TargetTable           string   `json:"target_table"`
	LineageType           string   `json:"lineage_type"`           // "direct", "transformed", "aggregated"
	SemanticSimilarity    float64  `json:"semantic_similarity"`    // Semantic similarity score
	TransformationPattern string   `json:"transformation_pattern"` // Detected transformation type
	DataFlowConfidence    float64  `json:"data_flow_confidence"`   // Confidence in lineage
	ContextualClues       []string `json:"contextual_clues"`       // Clues that led to lineage detection
}

// AnalyzeColumnSemantics performs semantic analysis of a column.
// Phase 8.1: Enhanced with domain-aware analysis using domain detector.
func (ssa *SemanticSchemaAnalyzer) AnalyzeColumnSemantics(
	ctx context.Context,
	columnName string,
	columnType string,
	tableName string,
	tableContext map[string]any,
	domainID string, // Phase 8.1: Optional domain ID for domain-aware analysis
) (*SemanticColumnAnalysis, error) {
	analysis := &SemanticColumnAnalysis{
		ColumnName:         columnName,
		TableName:          tableName,
		SemanticSimilarity: make(map[string]float64),
		NamingConventions:  []string{},
		ContextualFeatures: make(map[string]any),
	}

	// Phase 10: Use LNN-based terminology learning if available, otherwise fallback to fixed dictionaries
	if ssa.terminologyLearner != nil {
		// Use LNN for naming conventions
		analysis.NamingConventions = ssa.terminologyLearner.AnalyzeNamingConvention(ctx, columnName)

		// Use LNN for domain inference
		domain, domainConf := ssa.terminologyLearner.InferDomain(ctx, columnName, tableName, tableContext)
		// Phase 8.1: enhance with domain detector if domainID provided
		if domainID != "" && ssa.domainDetector != nil {
			enhancedDomain, enhancedConf := ssa.inferDomainWithKeywords(columnName, tableName, domainID)
			if enhancedConf > domainConf {
				domain = enhancedDomain
				domainConf = enhancedConf
			}
		}
		analysis.InferredDomain = domain
		analysis.DomainConfidence = domainConf

		// Use LNN for role inference
		role, roleConf := ssa.terminologyLearner.InferRole(ctx, columnName, columnType, tableName, tableContext)
		analysis.InferredBusinessRole = role
		analysis.BusinessRoleConfidence = roleConf
	} else {
		// Fallback to fixed dictionaries
		analysis.NamingConventions = ssa.analyzeNamingConventions(columnName, tableName)
		domain, domainConf := ssa.inferBusinessDomain(columnName, tableName, tableContext)
		// Phase 8.1: enhance with domain detector if domainID provided
		if domainID != "" && ssa.domainDetector != nil {
			enhancedDomain, enhancedConf := ssa.inferDomainWithKeywords(columnName, tableName, domainID)
			if enhancedConf > domainConf {
				domain = enhancedDomain
				domainConf = enhancedConf
			}
		}
		analysis.InferredDomain = domain
		analysis.DomainConfidence = domainConf
		role, roleConf := ssa.inferBusinessRole(columnName, columnType, tableName, tableContext)
		analysis.InferredBusinessRole = role
		analysis.BusinessRoleConfidence = roleConf
	}

	// Generate semantic embeddings for similarity analysis
	// Phase 8.1: Use domain tags for enhanced similarity if domainID provided
	if ssa.useSAPRPTEmbeddings {
		var similarities map[string]float64
		var err error

		if domainID != "" && ssa.domainDetector != nil {
			similarities, err = ssa.calculateSemanticSimilarityWithDomain(ctx, columnName, tableName, domainID)
		} else {
			similarities, err = ssa.calculateSemanticSimilarity(ctx, columnName, tableName)
		}

		if err == nil {
			analysis.SemanticSimilarity = similarities
		}
	}

	// Extract contextual features
	analysis.ContextualFeatures = ssa.extractContextualFeatures(columnName, columnType, tableName, tableContext)

	return analysis, nil
}

// AnalyzeDataLineage performs semantic analysis of data lineage.
func (ssa *SemanticSchemaAnalyzer) AnalyzeDataLineage(
	ctx context.Context,
	sourceTable string,
	targetTable string,
	sourceColumns []string,
	targetColumns []string,
	sqlQuery string,
) (*SemanticLineageAnalysis, error) {
	analysis := &SemanticLineageAnalysis{
		SourceTable:     sourceTable,
		TargetTable:     targetTable,
		ContextualClues: []string{},
	}

	// Analyze semantic similarity between source and target
	if ssa.useSAPRPTEmbeddings {
		similarity, err := ssa.calculateTableSimilarity(ctx, sourceTable, targetTable)
		if err == nil {
			analysis.SemanticSimilarity = similarity
		}
	}

	// Detect lineage type
	lineageType, confidence := ssa.detectLineageType(sourceTable, targetTable, sqlQuery)
	analysis.LineageType = lineageType
	analysis.DataFlowConfidence = confidence

	// Detect transformation pattern
	transformationPattern := ssa.detectTransformationPattern(sourceColumns, targetColumns, sqlQuery)
	analysis.TransformationPattern = transformationPattern

	// Extract contextual clues
	analysis.ContextualClues = ssa.extractLineageClues(sourceTable, targetTable, sqlQuery)

	return analysis, nil
}

// analyzeNamingConventions analyzes naming patterns in column and table names.
func (ssa *SemanticSchemaAnalyzer) analyzeNamingConventions(columnName, tableName string) []string {
	conventions := []string{}

	// Check for common naming patterns
	patterns := map[string]*regexp.Regexp{
		"snake_case":      regexp.MustCompile(`^[a-z]+(_[a-z]+)*$`),
		"camelCase":       regexp.MustCompile(`^[a-z]+[A-Z][a-zA-Z]*$`),
		"PascalCase":      regexp.MustCompile(`^[A-Z][a-zA-Z]*$`),
		"UPPER_SNAKE":     regexp.MustCompile(`^[A-Z]+(_[A-Z]+)*$`),
		"has_id_suffix":   regexp.MustCompile(`_id$`),
		"has_id_prefix":   regexp.MustCompile(`^id_`),
		"has_date_suffix": regexp.MustCompile(`_date$`),
		"has_ts_suffix":   regexp.MustCompile(`_ts$|_timestamp$`),
		"has_amount":      regexp.MustCompile(`amount|price|cost|value|total`),
		"has_status":      regexp.MustCompile(`status|state|flag`),
	}

	for patternName, pattern := range patterns {
		if pattern.MatchString(strings.ToLower(columnName)) {
			conventions = append(conventions, patternName)
		}
	}

	return conventions
}

// inferBusinessDomain infers the business domain from naming conventions and context.
func (ssa *SemanticSchemaAnalyzer) inferBusinessDomain(
	columnName string,
	tableName string,
	context map[string]any,
) (string, float64) {
	// Domain keywords mapping
	domainKeywords := map[string][]string{
		"financial": {"amount", "price", "cost", "revenue", "payment", "transaction", "account", "balance", "currency", "invoice"},
		"customer":  {"customer", "client", "user", "person", "contact", "email", "phone", "address"},
		"product":   {"product", "item", "sku", "catalog", "inventory", "stock", "quantity"},
		"order":     {"order", "order_item", "purchase", "cart", "basket"},
		"logistics": {"shipment", "delivery", "warehouse", "location", "address", "tracking"},
		"marketing": {"campaign", "promotion", "discount", "coupon", "advertisement"},
		"hr":        {"employee", "department", "position", "salary", "benefit"},
		"time":      {"date", "time", "timestamp", "period", "duration", "schedule"},
	}

	text := strings.ToLower(columnName + " " + tableName)
	maxScore := 0.0
	inferredDomain := "unknown"
	confidence := 0.0

	for domain, keywords := range domainKeywords {
		score := 0.0
		for _, keyword := range keywords {
			if strings.Contains(text, keyword) {
				score += 1.0
			}
		}
		normalizedScore := score / float64(len(keywords))
		if normalizedScore > maxScore {
			maxScore = normalizedScore
			inferredDomain = domain
			confidence = normalizedScore
		}
	}

	// Boost confidence if context provides additional clues
	if context != nil {
		if contextDomain, ok := context["domain"].(string); ok && contextDomain == inferredDomain {
			confidence = min(1.0, confidence+0.2)
		}
	}

	return inferredDomain, confidence
}

// inferBusinessRole infers the business role of a column.
func (ssa *SemanticSchemaAnalyzer) inferBusinessRole(
	columnName string,
	columnType string,
	tableName string,
	context map[string]any,
) (string, float64) {
	// Role keywords mapping
	roleKeywords := map[string][]string{
		"identifier": {"id", "key", "code", "number", "ref", "reference"},
		"amount":     {"amount", "price", "cost", "revenue", "value", "total", "sum", "balance"},
		"date":       {"date", "time", "timestamp", "created", "updated", "modified", "when"},
		"status":     {"status", "state", "flag", "active", "enabled", "deleted"},
		"name":       {"name", "title", "label", "description", "text"},
		"email":      {"email", "mail", "e_mail"},
		"phone":      {"phone", "tel", "telephone", "mobile"},
		"address":    {"address", "street", "city", "zip", "postal", "country"},
		"quantity":   {"quantity", "qty", "count", "number", "num"},
	}

	text := strings.ToLower(columnName)
	maxScore := 0.0
	inferredRole := "unknown"
	confidence := 0.0

	for role, keywords := range roleKeywords {
		score := 0.0
		for _, keyword := range keywords {
			if strings.Contains(text, keyword) {
				score += 1.0
			}
		}
		normalizedScore := score / float64(len(keywords))
		if normalizedScore > maxScore {
			maxScore = normalizedScore
			inferredRole = role
			confidence = normalizedScore
		}
	}

	// Boost confidence based on column type
	if columnType != "" {
		typeRoleMap := map[string]string{
			"int":       "identifier",
			"bigint":    "identifier",
			"decimal":   "amount",
			"numeric":   "amount",
			"date":      "date",
			"timestamp": "date",
			"varchar":   "name",
			"text":      "name",
			"boolean":   "status",
		}
		if expectedRole, ok := typeRoleMap[strings.ToLower(columnType)]; ok && expectedRole == inferredRole {
			confidence = min(1.0, confidence+0.3)
		}
	}

	return inferredRole, confidence
}

// inferDomainWithKeywords infers domain using domain detector keywords.
// Phase 8.1: Enhanced domain inference using domain config keywords.
func (ssa *SemanticSchemaAnalyzer) inferDomainWithKeywords(
	columnName string,
	tableName string,
	domainID string,
) (string, float64) {
	if ssa.domainDetector == nil {
		return "unknown", 0.0
	}

	// Get domain config from detector
	domainConfig, exists := ssa.domainDetector.Config(domainID)

	if !exists {
		return "unknown", 0.0
	}

	text := strings.ToLower(columnName + " " + tableName)
	domainKeywords := domainConfig.Keywords

	// Calculate keyword match score
	matches := 0
	for _, keyword := range domainKeywords {
		if strings.Contains(text, strings.ToLower(keyword)) {
			matches++
		}
	}

	if len(domainKeywords) == 0 {
		return domainConfig.Name, 0.5 // Default confidence if no keywords
	}

	confidence := float64(matches) / float64(len(domainKeywords))
	if confidence > 0.5 {
		return domainConfig.Name, min(1.0, confidence)
	}

	// Return detected domain but with lower confidence
	return domainConfig.Name, confidence
}

// calculateSemanticSimilarityWithDomain calculates semantic similarity using domain context.
// Phase 8.1: Enhanced similarity calculation with domain tags.
func (ssa *SemanticSchemaAnalyzer) calculateSemanticSimilarityWithDomain(
	ctx context.Context,
	columnName string,
	tableName string,
	domainID string,
) (map[string]float64, error) {
	similarities := make(map[string]float64)

	if ssa.domainDetector == nil {
		// Fallback to regular similarity
		return ssa.calculateSemanticSimilarity(ctx, columnName, tableName)
	}

	// Get domain config
	domainConfig, exists := ssa.domainDetector.Config(domainID)

	if !exists {
		// Fallback to regular similarity
		return ssa.calculateSemanticSimilarity(ctx, columnName, tableName)
	}

	// Use domain tags as known patterns
	domainTags := domainConfig.Tags
	if len(domainTags) == 0 {
		domainTags = domainConfig.Keywords // Fallback to keywords
	}

	// Generate embedding for column/table
	query := fmt.Sprintf("%s %s", columnName, tableName)
	cmd := exec.CommandContext(ctx, "python3", "./scripts/embed_sap_rpt.py",
		"--artifact-type", "text",
		"--text", query,
	)

	output, err := cmd.Output()
	if err != nil {
		return similarities, fmt.Errorf("failed to generate embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return similarities, fmt.Errorf("failed to unmarshal embedding: %w", err)
	}

	// Calculate similarity to domain tags
	for _, tag := range domainTags {
		tagCmd := exec.CommandContext(ctx, "python3", "./scripts/embed_sap_rpt.py",
			"--artifact-type", "text",
			"--text", tag,
		)

		tagOutput, err := tagCmd.Output()
		if err != nil {
			continue
		}

		var tagEmbedding []float32
		if err := json.Unmarshal(tagOutput, &tagEmbedding); err != nil {
			continue
		}

		similarity := cosineSimilarity(embedding, tagEmbedding)
		similarities[tag] = similarity
	}

	return similarities, nil
}

// calculateSemanticSimilarity calculates semantic similarity to known patterns.
func (ssa *SemanticSchemaAnalyzer) calculateSemanticSimilarity(
	ctx context.Context,
	columnName string,
	tableName string,
) (map[string]float64, error) {
	similarities := make(map[string]float64)

	// Use semantic embedding to find similar columns
	if ssa.extractServiceURL != "" {
		// Query the Extract service for similar columns
		query := fmt.Sprintf("%s %s", columnName, tableName)

		cmd := exec.CommandContext(ctx, "python3", "./scripts/embed_sap_rpt.py",
			"--artifact-type", "text",
			"--text", query,
		)

		output, err := cmd.Output()
		if err != nil {
			return similarities, fmt.Errorf("failed to generate embedding: %w", err)
		}

		var embedding []float32
		if err := json.Unmarshal(output, &embedding); err != nil {
			return similarities, fmt.Errorf("failed to unmarshal embedding: %w", err)
		}

		// Known patterns (would be loaded from knowledge base in production)
		knownPatterns := map[string]string{
			"customer_id":  "identifier customer",
			"order_amount": "amount financial",
			"created_date": "date time",
			"status_flag":  "status boolean",
			"product_name": "name product",
		}

		// Calculate similarity to known patterns
		for patternName, patternText := range knownPatterns {
			patternCmd := exec.CommandContext(ctx, "python3", "./scripts/embed_sap_rpt.py",
				"--artifact-type", "text",
				"--text", patternText,
			)

			patternOutput, err := patternCmd.Output()
			if err != nil {
				continue
			}

			var patternEmbedding []float32
			if err := json.Unmarshal(patternOutput, &patternEmbedding); err != nil {
				continue
			}

			// Calculate cosine similarity
			similarity := cosineSimilarity(embedding, patternEmbedding)
			similarities[patternName] = similarity
		}
	}

	return similarities, nil
}

// calculateTableSimilarity calculates semantic similarity between two tables.
func (ssa *SemanticSchemaAnalyzer) calculateTableSimilarity(
	ctx context.Context,
	sourceTable string,
	targetTable string,
) (float64, error) {
	// Generate embeddings for both tables
	sourceCmd := exec.CommandContext(ctx, "python3", "./scripts/embed_sap_rpt.py",
		"--artifact-type", "text",
		"--text", sourceTable,
	)

	sourceOutput, err := sourceCmd.Output()
	if err != nil {
		return 0.0, fmt.Errorf("failed to generate source embedding: %w", err)
	}

	var sourceEmbedding []float32
	if err := json.Unmarshal(sourceOutput, &sourceEmbedding); err != nil {
		return 0.0, fmt.Errorf("failed to unmarshal source embedding: %w", err)
	}

	targetCmd := exec.CommandContext(ctx, "python3", "./scripts/embed_sap_rpt.py",
		"--artifact-type", "text",
		"--text", targetTable,
	)

	targetOutput, err := targetCmd.Output()
	if err != nil {
		return 0.0, fmt.Errorf("failed to generate target embedding: %w", err)
	}

	var targetEmbedding []float32
	if err := json.Unmarshal(targetOutput, &targetEmbedding); err != nil {
		return 0.0, fmt.Errorf("failed to unmarshal target embedding: %w", err)
	}

	// Calculate cosine similarity
	similarity := cosineSimilarity(sourceEmbedding, targetEmbedding)
	return similarity, nil
}

// detectLineageType detects the type of data lineage.
func (ssa *SemanticSchemaAnalyzer) detectLineageType(
	sourceTable string,
	targetTable string,
	sqlQuery string,
) (string, float64) {
	sqlUpper := strings.ToUpper(sqlQuery)

	// Direct copy (no transformation)
	if strings.Contains(sqlUpper, "INSERT INTO") && strings.Contains(sqlUpper, "SELECT *") {
		return "direct", 0.9
	}

	// Aggregation
	if strings.Contains(sqlUpper, "GROUP BY") || strings.Contains(sqlUpper, "SUM(") ||
		strings.Contains(sqlUpper, "COUNT(") || strings.Contains(sqlUpper, "AVG(") {
		return "aggregated", 0.85
	}

	// Transformation (JOIN, CASE, etc.)
	if strings.Contains(sqlUpper, "JOIN") || strings.Contains(sqlUpper, "CASE") ||
		strings.Contains(sqlUpper, "CAST") || strings.Contains(sqlUpper, "CONVERT") {
		return "transformed", 0.8
	}

	// Default
	return "direct", 0.5
}

// detectTransformationPattern detects the type of transformation.
func (ssa *SemanticSchemaAnalyzer) detectTransformationPattern(
	sourceColumns []string,
	targetColumns []string,
	sqlQuery string,
) string {
	sqlUpper := strings.ToUpper(sqlQuery)

	if strings.Contains(sqlUpper, "SUM(") || strings.Contains(sqlUpper, "AVG(") {
		return "aggregation"
	}
	if strings.Contains(sqlUpper, "JOIN") {
		return "join"
	}
	if strings.Contains(sqlUpper, "CASE") {
		return "conditional"
	}
	if strings.Contains(sqlUpper, "CAST") || strings.Contains(sqlUpper, "CONVERT") {
		return "type_conversion"
	}
	if len(sourceColumns) != len(targetColumns) {
		return "column_mapping"
	}

	return "direct_copy"
}

// extractContextualFeatures extracts context-aware features.
func (ssa *SemanticSchemaAnalyzer) extractContextualFeatures(
	columnName string,
	columnType string,
	tableName string,
	context map[string]any,
) map[string]any {
	features := make(map[string]any)

	features["column_name_length"] = len(columnName)
	features["table_name_length"] = len(tableName)
	features["has_underscore"] = strings.Contains(columnName, "_")
	features["word_count"] = len(strings.Fields(columnName))
	features["column_type"] = columnType

	// Add context features if available
	if context != nil {
		if schema, ok := context["schema"].(string); ok {
			features["schema"] = schema
		}
		if database, ok := context["database"].(string); ok {
			features["database"] = database
		}
	}

	return features
}

// extractLineageClues extracts clues that led to lineage detection.
func (ssa *SemanticSchemaAnalyzer) extractLineageClues(
	sourceTable string,
	targetTable string,
	sqlQuery string,
) []string {
	clues := []string{}

	sqlUpper := strings.ToUpper(sqlQuery)

	// Check for direct table references
	if strings.Contains(sqlUpper, strings.ToUpper(sourceTable)) {
		clues = append(clues, "source_table_referenced")
	}
	if strings.Contains(sqlUpper, strings.ToUpper(targetTable)) {
		clues = append(clues, "target_table_referenced")
	}

	// Check for INSERT INTO pattern
	if strings.Contains(sqlUpper, "INSERT INTO") && strings.Contains(sqlUpper, strings.ToUpper(targetTable)) {
		clues = append(clues, "insert_into_target")
	}

	// Check for SELECT FROM pattern
	if strings.Contains(sqlUpper, "SELECT") && strings.Contains(sqlUpper, "FROM") {
		clues = append(clues, "select_from_source")
	}

	return clues
}
