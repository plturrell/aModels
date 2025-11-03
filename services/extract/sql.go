package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/Chahine-tech/sql-parser-go/pkg/parser"
)

type ColumnLineage struct {
	SourceColumn string `json:"source_column"`
	TargetColumn string `json:"target_column"`
	SourceTable  string `json:"source_table,omitempty"`
	TargetTable  string `json:"target_table,omitempty"`
}

type Lineage struct {
	SourceTables  []string        `json:"source_tables"`
	TargetTables  []string        `json:"target_tables"`
	ColumnLineage []ColumnLineage `json:"column_lineage"`
}

func parseSQL(sql string) (*Lineage, error) {
	p := parser.New(sql)
	ast, err := p.Parse()
	if err != nil {
		if fallback := fallbackParseSQL(sql); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("failed to parse sql: %w", err)
	}

	builder := newLineageBuilder()
	builder.extractLineage(ast, lineageContext{})
	result := builder.Build()

	if fallback := fallbackParseSQL(sql); fallback != nil {
		result.SourceTables = mergeIdentifierLists(result.SourceTables, fallback.SourceTables)
		result.TargetTables = mergeIdentifierLists(result.TargetTables, fallback.TargetTables)
		if len(result.ColumnLineage) == 0 && len(fallback.ColumnLineage) > 0 {
			result.ColumnLineage = append([]ColumnLineage(nil), fallback.ColumnLineage...)
		}
	}

	return result, nil
}

type lineageContext struct {
	targetTables []string
}

type lineageBuilder struct {
	sourceSeen map[string]string
	targetSeen map[string]string
	sourceList []string
	targetList []string

	columnSeen map[string]struct{}
	columns    []ColumnLineage
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

func (b *lineageBuilder) addColumnLineage(sourceCol, targetCol, sourceTable, targetTable string) {
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
	b.columns = append(b.columns, ColumnLineage{
		SourceColumn: sourceCol,
		TargetColumn: targetCol,
		SourceTable:  sourceTable,
		TargetTable:  targetTable,
	})
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

			b.addColumnLineage(source, target, sourceTable, targetTable)
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

func shortIdentifier(name string) string {
	norm := normalizeIdentifier(name)
	if norm == "" {
		return ""
	}
	parts := strings.Split(norm, ".")
	return strings.ToLower(parts[len(parts)-1])
}
