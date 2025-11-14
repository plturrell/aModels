package storage

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/Chahine-tech/sql-parser-go/pkg/parser"
)

type ColumnLineage struct {
	SourceColumn      string `json:"source_column"`
	TargetColumn      string `json:"target_column"`
	SourceTable       string `json:"source_table,omitempty"`
	TargetTable       string `json:"target_table,omitempty"`
	TransformationType string `json:"transformation_type,omitempty"`
	SQLExpression     string `json:"sql_expression,omitempty"`
	Function          string `json:"function,omitempty"`
	JoinType          string `json:"join_type,omitempty"`
	JoinCondition     string `json:"join_condition,omitempty"`
	FilterCondition   string `json:"filter_condition,omitempty"`
	AggregationKeys   []string `json:"aggregation_keys,omitempty"`
}

type Lineage struct {
	SourceTables  []string        `json:"source_tables"`
	TargetTables  []string        `json:"target_tables"`
	ColumnLineage []ColumnLineage `json:"column_lineage"`
}

// ParseSQL parses a SQL query and extracts lineage information.
func ParseSQL(sql string) (*Lineage, error) {
	p := parser.New(sql)
	ast, err := p.ParseStatement()
	if err != nil {
		if fallback := fallbackParseSQL(sql); fallback != nil {
			// Enhance fallback with basic transformation detection
			enhanceFallbackLineage(fallback, sql)
			return fallback, nil
		}
		return nil, fmt.Errorf("failed to parse sql: %w", err)
	}

	// Convert Statement AST to map format for extractLineage
	astMap := statementToMap(ast)

	builder := newLineageBuilder()
	builder.sqlQuery = sql
	ctx := lineageContext{
		targetTables:   []string{},
		joinConditions: make(map[string]string),
		joinTypes:      make(map[string]string),
		currentSQL:     sql,
	}
	builder.extractLineage(astMap, ctx)
	result := builder.Build()

	if fallback := fallbackParseSQL(sql); fallback != nil {
		result.SourceTables = mergeIdentifierLists(result.SourceTables, fallback.SourceTables)
		result.TargetTables = mergeIdentifierLists(result.TargetTables, fallback.TargetTables)
		if len(result.ColumnLineage) == 0 && len(fallback.ColumnLineage) > 0 {
			enhanceFallbackLineage(fallback, sql)
			result.ColumnLineage = append([]ColumnLineage(nil), fallback.ColumnLineage...)
		}
	}

	return result, nil
}

type lineageContext struct {
	targetTables     []string
	whereCondition   string
	groupByColumns   []string
	joinConditions   map[string]string // table -> condition
	joinTypes        map[string]string // table -> join type
	currentSQL       string
}

type lineageBuilder struct {
	sourceSeen map[string]string
	targetSeen map[string]string
	sourceList []string
	targetList []string

	columnSeen map[string]struct{}
	columns    []ColumnLineage
	sqlQuery   string
}

func newLineageBuilder() *lineageBuilder {
	return &lineageBuilder{
		sourceSeen: map[string]string{},
		targetSeen: map[string]string{},
		columnSeen: map[string]struct{}{},
	}
}

func (b *lineageBuilder) Build() *Lineage {
	return &Lineage{
		SourceTables:  append([]string(nil), b.sourceList...),
		TargetTables:  append([]string(nil), b.targetList...),
		ColumnLineage: append([]ColumnLineage(nil), b.columns...),
	}
}

func (b *lineageBuilder) addSourceTable(name string) {
	name = normalizeIdentifier(name)
	if name == "" {
		return
	}
	key := strings.ToLower(name)
	if _, ok := b.sourceSeen[key]; ok {
		return
	}
	b.sourceSeen[key] = name
	b.sourceList = append(b.sourceList, name)
}

func (b *lineageBuilder) addTargetTable(name string) {
	name = normalizeIdentifier(name)
	if name == "" {
		return
	}
	key := strings.ToLower(name)
	if _, ok := b.targetSeen[key]; ok {
		return
	}
	b.targetSeen[key] = name
	b.targetList = append(b.targetList, name)
}

func (b *lineageBuilder) addColumnLineage(sourceCol, targetCol, sourceTable, targetTable string, ctx lineageContext) {
	sourceCol = normalizeIdentifier(sourceCol)
	targetCol = normalizeIdentifier(targetCol)
	if sourceCol == "" || targetCol == "" {
		return
	}

	sourceTable = normalizeIdentifier(sourceTable)
	targetTable = normalizeIdentifier(targetTable)

	key := strings.ToLower(fmt.Sprintf("%s|%s|%s|%s", sourceTable, sourceCol, targetTable, targetCol))
	if _, exists := b.columnSeen[key]; exists {
		return
	}
	b.columnSeen[key] = struct{}{}

	// Detect transformation type and extract details
	transformationType, function, sqlExpr := detectTransformation(b.sqlQuery, sourceCol, targetCol)
	joinType := ""
	joinCondition := ""
	if sourceTable != "" {
		if jt, ok := ctx.joinTypes[sourceTable]; ok {
			joinType = jt
		}
		if jc, ok := ctx.joinConditions[sourceTable]; ok {
			joinCondition = jc
		}
	}

	lineage := ColumnLineage{
		SourceColumn:      sourceCol,
		TargetColumn:      targetCol,
		SourceTable:       sourceTable,
		TargetTable:       targetTable,
		TransformationType: transformationType,
		SQLExpression:     sqlExpr,
		Function:          function,
		JoinType:          joinType,
		JoinCondition:     joinCondition,
		FilterCondition:   ctx.whereCondition,
		AggregationKeys:   ctx.groupByColumns,
	}

	b.columns = append(b.columns, lineage)
}

func (b *lineageBuilder) extractLineage(node map[string]interface{}, ctx lineageContext) {
	if node == nil {
		return
	}

	if _, ok := node["insert"]; ok {
		targets := b.collectTables(node["tables"], b.addTargetTable)
		ctx.targetTables = append([]string(nil), targets...)

		if selectStmt, ok := node["select"].(map[string]interface{}); ok {
			b.collectTables(selectStmt, b.addSourceTable)
			b.extractColumnLineage(selectStmt, ctx)
			b.extractLineage(selectStmt, ctx)
		}
		return
	}

	if _, ok := node["update"]; ok {
		targets := b.collectTables(node["tables"], b.addTargetTable)
		ctx.targetTables = append([]string(nil), targets...)
		b.collectTables(node["where"], b.addSourceTable)
		b.recurse(node["where"], ctx)
		return
	}

	if _, ok := node["create"]; ok {
		if table, ok := node["table"].(string); ok {
			b.addTargetTable(table)
			ctx.targetTables = []string{normalizeIdentifier(table)}
		}
		if selectStmt, ok := node["select"].(map[string]interface{}); ok {
			b.collectTables(selectStmt, b.addSourceTable)
			b.extractColumnLineage(selectStmt, ctx)
			b.extractLineage(selectStmt, ctx)
		}
		return
	}

	for _, value := range node {
		b.recurse(value, ctx)
	}
}

func (b *lineageBuilder) recurse(value interface{}, ctx lineageContext) {
	switch v := value.(type) {
	case map[string]interface{}:
		b.extractLineage(v, ctx)
	case []interface{}:
		for _, item := range v {
			b.recurse(item, ctx)
		}
	}
}

func (b *lineageBuilder) collectTables(value interface{}, add func(string)) []string {
	collected := []string{}
	if value == nil {
		return collected
	}

	record := func(name string) {
		name = normalizeIdentifier(name)
		if name == "" {
			return
		}
		collected = append(collected, name)
		if add != nil {
			add(name)
		}
	}

	switch v := value.(type) {
	case string:
		record(v)
	case []interface{}:
		for _, item := range v {
			collected = append(collected, b.collectTables(item, add)...)
		}
	case map[string]interface{}:
		for key, val := range v {
			switch strings.ToLower(key) {
			case "from", "tables", "join", "into", "update":
				collected = append(collected, b.collectTables(val, add)...)
			case "table":
				collected = append(collected, b.collectTables(val, add)...)
			case "select", "where", "with":
				collected = append(collected, b.collectTables(val, add)...)
			}
		}
	}

	return collected
}

func (b *lineageBuilder) extractColumnLineage(node map[string]interface{}, ctx lineageContext) {
	if node == nil {
		return
	}

	if columns, ok := node["columns"].([]interface{}); ok {
		for _, column := range columns {
			colMap, ok := column.(map[string]interface{})
			if !ok {
				continue
			}

			var source, target, sourceTable string
			if expr, ok := colMap["expr"].(map[string]interface{}); ok {
				if col, ok := expr["column"].(string); ok {
					source = col
				}
				if tbl, ok := expr["table"].(string); ok {
					sourceTable = tbl
				}
			}

			if as, ok := colMap["as"].(string); ok {
				target = as
			} else {
				target = source
			}

			targetTable := ""
			if len(ctx.targetTables) == 1 {
				targetTable = ctx.targetTables[0]
			}

			b.addColumnLineage(source, target, sourceTable, targetTable, ctx)
		}
	}

	for _, value := range node {
		b.recurse(value, ctx)
	}
}

func normalizeIdentifier(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	value = strings.Trim(value, "`\"")
	value = strings.TrimSuffix(value, ";")
	return value
}

var (
	insertTargetRegex = regexp.MustCompile(`(?i)insert\s+into\s+([a-z0-9_."$]+)`)
	createTargetRegex = regexp.MustCompile(`(?i)create\s+table\s+(?:if\s+not\s+exists\s+)?([a-z0-9_."$]+)`)
	fromSourceRegex   = regexp.MustCompile(`(?i)from\s+([a-z0-9_."$]+)`)
	joinSourceRegex   = regexp.MustCompile(`(?i)join\s+([a-z0-9_."$]+)`)
	selectColumnRegex = regexp.MustCompile(`(?is)select\s+(.*?)\s+from`)
)

func fallbackParseSQL(sql string) *Lineage {
	targets := extractMatches(insertTargetRegex, sql)
	if len(targets) == 0 {
		targets = extractMatches(createTargetRegex, sql)
	}

	sources := extractMatches(fromSourceRegex, sql)
	sources = append(sources, extractMatches(joinSourceRegex, sql)...)

	if len(targets) == 0 && len(sources) == 0 {
		return nil
	}

	sourceUnique := uniqueIdentifiers(sources)
	targetUnique := uniqueIdentifiers(targets)

	lineage := &Lineage{
		SourceTables: sourceUnique,
		TargetTables: targetUnique,
	}

	if len(targetUnique) == 1 {
		if matches := selectColumnRegex.FindStringSubmatch(sql); len(matches) == 2 {
			for _, col := range splitColumns(matches[1]) {
				col = normalizeIdentifier(col)
				if col == "" {
					continue
				}
				lineage.ColumnLineage = append(lineage.ColumnLineage, ColumnLineage{
					SourceColumn: col,
					TargetColumn: col,
					TargetTable:  targetUnique[0],
				})
			}
		}
	}

	return lineage
}

func extractMatches(re *regexp.Regexp, sql string) []string {
	matches := re.FindAllStringSubmatch(sql, -1)
	if len(matches) == 0 {
		return nil
	}
	results := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			results = append(results, match[1])
		}
	}
	return results
}

func uniqueIdentifiers(values []string) []string {
	seen := map[string]string{}
	ordered := []string{}
	for _, v := range values {
		norm := normalizeIdentifier(v)
		if norm == "" {
			continue
		}
		key := strings.ToLower(norm)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = norm
		ordered = append(ordered, norm)
	}
	return ordered
}

func mergeIdentifierLists(primary, supplemental []string) []string {
	if len(primary) == 0 {
		return uniqueIdentifiers(supplemental)
	}

	order := []string{}
	seenFull := map[string]struct{}{}
	preferred := map[string]string{}

	add := func(name string) {
		norm := normalizeIdentifier(name)
		if norm == "" {
			return
		}

		fullKey := strings.ToLower(norm)
		if _, ok := seenFull[fullKey]; ok {
			return
		}

		shortKey := shortIdentifier(norm)
		if existing, ok := preferred[shortKey]; ok {
			existsHasSchema := strings.Contains(existing, ".")
			currentHasSchema := strings.Contains(norm, ".")
			if currentHasSchema && !existsHasSchema {
				for i, v := range order {
					if strings.EqualFold(v, existing) {
						order[i] = norm
						delete(seenFull, strings.ToLower(existing))
						break
					}
				}
				seenFull[fullKey] = struct{}{}
				preferred[shortKey] = norm
				return
			}
			if existsHasSchema && !currentHasSchema {
				return
			}
		}

		order = append(order, norm)
		seenFull[fullKey] = struct{}{}
		if _, ok := preferred[shortKey]; !ok || strings.Contains(norm, ".") {
			preferred[shortKey] = norm
		}
	}

	for _, v := range supplemental {
		add(v)
	}
	for _, v := range primary {
		add(v)
	}

	return order
}

func splitColumns(value string) []string {
	var (
		columns []string
		buf     strings.Builder
		depth   int
	)

	flush := func() {
		part := strings.TrimSpace(buf.String())
		if part != "" {
			columns = append(columns, stripAlias(part))
		}
		buf.Reset()
	}

	for _, r := range value {
		switch r {
		case '(', '[', '{':
			depth++
		case ')', ']', '}':
			if depth > 0 {
				depth--
			}
		case ',':
			if depth == 0 {
				flush()
				continue
			}
		}
		buf.WriteRune(r)
	}
	flush()

	return columns
}

func stripAlias(fragment string) string {
	frag := strings.TrimSpace(fragment)
	lower := strings.ToLower(frag)
	if idx := strings.LastIndex(lower, " as "); idx != -1 {
		return strings.TrimSpace(frag[idx+4:])
	}

	parts := strings.Fields(frag)
	if len(parts) >= 2 {
		return parts[len(parts)-1]
	}
	return frag
}

// statementToMap converts a parser.Statement AST to a map format expected by extractLineage
func statementToMap(stmt parser.Statement) map[string]interface{} {
	result := make(map[string]interface{})
	
	switch s := stmt.(type) {
	case *parser.SelectStatement:
		result["select"] = selectStatementToMap(s)
	case *parser.InsertStatement:
		result["insert"] = true
		result["tables"] = []interface{}{s.Table.String()}
	case *parser.UpdateStatement:
		result["update"] = true
		result["tables"] = []interface{}{s.Table.String()}
		if s.Where != nil {
			result["where"] = expressionToMap(s.Where)
		}
	case *parser.DeleteStatement:
		result["delete"] = true
		// DeleteStatement may not have a From field - check structure
		if s.Where != nil {
			result["where"] = expressionToMap(s.Where)
		}
	}
	
	return result
}

func selectStatementToMap(s *parser.SelectStatement) map[string]interface{} {
	result := make(map[string]interface{})
	
	if s.From != nil {
		tables := []interface{}{}
		for _, table := range s.From.Tables {
			tables = append(tables, table.String())
		}
		for _, join := range s.Joins {
			if join.Table.Name != "" {
				tables = append(tables, join.Table.String())
			}
		}
		result["tables"] = tables
	}
	
	if len(s.Columns) > 0 {
		columns := []interface{}{}
		for _, col := range s.Columns {
			columns = append(columns, col.String())
		}
		result["columns"] = columns
	}
	
	return result
}

func expressionToMap(expr parser.Expression) map[string]interface{} {
	result := make(map[string]interface{})
	result["type"] = expr.Type()
	result["value"] = expr.String()
	return result
}

func shortIdentifier(name string) string {
	norm := normalizeIdentifier(name)
	if norm == "" {
		return ""
	}
	parts := strings.Split(norm, ".")
	return strings.ToLower(parts[len(parts)-1])
}

// detectTransformation analyzes SQL to detect transformation type, function, and expression
func detectTransformation(sql, sourceCol, targetCol string) (transformationType, function, sqlExpr string) {
	sqlUpper := strings.ToUpper(sql)
	
	// Check for aggregation functions
	aggFunctions := []string{"SUM", "COUNT", "AVG", "MAX", "MIN", "STDDEV", "VARIANCE"}
	for _, fn := range aggFunctions {
		pattern := fmt.Sprintf(`%s\s*\(`, fn)
		if matched, _ := regexp.MatchString(pattern, sqlUpper); matched {
			if strings.Contains(sqlUpper, "GROUP BY") {
				return "aggregation", fn, extractExpression(sql, targetCol)
			}
		}
	}
	
	// Check for CASE expressions
	if strings.Contains(sqlUpper, "CASE") && strings.Contains(sqlUpper, "WHEN") {
		return "conditional", "CASE", extractExpression(sql, targetCol)
	}
	
	// Check for JOIN
	if strings.Contains(sqlUpper, "JOIN") {
		return "join", "", extractExpression(sql, targetCol)
	}
	
	// Check for CAST/CONVERT
	if strings.Contains(sqlUpper, "CAST") || strings.Contains(sqlUpper, "CONVERT") {
		return "cast", "CAST", extractExpression(sql, targetCol)
	}
	
	// Check for WHERE (filter)
	if strings.Contains(sqlUpper, "WHERE") {
		return "filter", "", extractExpression(sql, targetCol)
	}
	
	// Default: direct copy
	if sourceCol == targetCol {
		return "direct_copy", "", sourceCol
	}
	
	return "transformed", "", extractExpression(sql, targetCol)
}

// extractExpression attempts to extract the SQL expression for a column
func extractExpression(sql, columnName string) string {
	// Try to find the column expression in SELECT clause
	selectPattern := regexp.MustCompile(`(?i)SELECT\s+(.*?)\s+FROM`)
	matches := selectPattern.FindStringSubmatch(sql)
	if len(matches) > 1 {
		selectClause := matches[1]
		// Look for the column in the SELECT clause
		colPattern := regexp.MustCompile(fmt.Sprintf(`(?i)([^,]+?)\s+(?:AS\s+)?%s\b`, regexp.QuoteMeta(columnName)))
		colMatches := colPattern.FindStringSubmatch(selectClause)
		if len(colMatches) > 1 {
			return strings.TrimSpace(colMatches[1])
		}
	}
	return columnName
}

// enhanceFallbackLineage adds transformation metadata to fallback lineage
func enhanceFallbackLineage(lineage *Lineage, sql string) {
	sqlUpper := strings.ToUpper(sql)
	
	for i := range lineage.ColumnLineage {
		cl := &lineage.ColumnLineage[i]
		
		// Detect transformation type
		transformationType, function, sqlExpr := detectTransformation(sql, cl.SourceColumn, cl.TargetColumn)
		cl.TransformationType = transformationType
		cl.Function = function
		cl.SQLExpression = sqlExpr
		
		// Extract WHERE condition
		if strings.Contains(sqlUpper, "WHERE") {
			wherePattern := regexp.MustCompile(`(?i)WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s*$)`)
			whereMatches := wherePattern.FindStringSubmatch(sql)
			if len(whereMatches) > 1 {
				cl.FilterCondition = strings.TrimSpace(whereMatches[1])
			}
		}
		
		// Extract GROUP BY
		if strings.Contains(sqlUpper, "GROUP BY") {
			groupByPattern := regexp.MustCompile(`(?i)GROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|\s*$)`)
			groupByMatches := groupByPattern.FindStringSubmatch(sql)
			if len(groupByMatches) > 1 {
				groupByClause := strings.TrimSpace(groupByMatches[1])
				cl.AggregationKeys = splitColumns(groupByClause)
			}
		}
		
		// Extract JOIN information
		joinPattern := regexp.MustCompile(`(?i)(LEFT|RIGHT|INNER|FULL|OUTER)?\s*JOIN\s+(\w+)\s+ON\s+(.+?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s*$)`)
		joinMatches := joinPattern.FindAllStringSubmatch(sql, -1)
		for _, match := range joinMatches {
			if len(match) >= 4 {
				joinType := strings.TrimSpace(match[1])
				if joinType == "" {
					joinType = "INNER"
				}
				table := strings.TrimSpace(match[2])
				condition := strings.TrimSpace(match[3])
				if cl.SourceTable == table {
					cl.JoinType = joinType
					cl.JoinCondition = condition
				}
			}
		}
	}
}
