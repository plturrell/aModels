# Proposed Improvements to Extract Service

## Code Changes Required

### 1. Add Orphan Column Fix to Normalization

**File**: `services/extract/normalization.go`

Add after line 100 (after edge normalization):

```go
// fixOrphanColumns creates missing HAS_COLUMN edges for columns that reference tables by ID pattern
func fixOrphanColumns(nodes []Node, edges []Edge) ([]Node, []Edge, []string) {
	var warnings []string
	
	// Build index of tables and columns
	tableMap := make(map[string]Node)
	columnMap := make(map[string]Node)
	for _, node := range nodes {
		if node.Type == "table" {
			tableMap[node.ID] = node
		} else if node.Type == "column" {
			columnMap[node.ID] = node
		}
	}
	
	// Build index of existing HAS_COLUMN edges
	existingEdges := make(map[string]bool)
	for _, edge := range edges {
		if edge.Label == "HAS_COLUMN" {
			key := fmt.Sprintf("%s->%s", edge.SourceID, edge.TargetID)
			existingEdges[key] = true
		}
	}
	
	// Find and fix orphan columns
	fixed := 0
	for colID, col := range columnMap {
		if colID == "" {
			continue
		}
		
		// Check if column already has a HAS_COLUMN edge
		hasEdge := false
		for _, edge := range edges {
			if edge.TargetID == colID && edge.Label == "HAS_COLUMN" {
				hasEdge = true
				break
			}
		}
		
		if hasEdge {
			continue
		}
		
		// Try to find parent table by ID pattern
		// Column IDs are typically: table.column or schema.table.column
		parts := strings.Split(colID, ".")
		if len(parts) < 2 {
			continue
		}
		
		// Try different table ID patterns
		var tableID string
		if len(parts) == 2 {
			// Simple: table.column
			tableID = parts[0]
		} else if len(parts) >= 3 {
			// Schema.table.column or table.column (if last part is column name)
			// Try: schema.table
			tableID = strings.Join(parts[:len(parts)-1], ".")
			// Also try: table (if schema.table doesn't match)
			if _, ok := tableMap[tableID]; !ok && len(parts) >= 3 {
				tableID = parts[len(parts)-2] // Second-to-last part
			}
		}
		
		if table, ok := tableMap[tableID]; ok {
			edgeKey := fmt.Sprintf("%s->%s", table.ID, colID)
			if !existingEdges[edgeKey] {
				edges = append(edges, Edge{
					SourceID: table.ID,
					TargetID: colID,
					Label:    "HAS_COLUMN",
				})
				fixed++
			}
		}
	}
	
	if fixed > 0 {
		warnings = append(warnings, fmt.Sprintf("fixed %d orphan columns by creating missing HAS_COLUMN edges", fixed))
	}
	
	return nodes, edges, warnings
}
```

Then call it in `normalizeGraph()`:
```go
// After edge normalization (around line 100)
nodes, edges, fixWarnings := fixOrphanColumns(nodes, edges)
result.Warnings = append(result.Warnings, fixWarnings...)
```

### 2. Improve Type Normalization

**File**: `services/extract/main.go`

Add function:
```go
func normalizeColumnType(rawType string) string {
	rawType = strings.ToLower(strings.TrimSpace(rawType))
	
	// Normalize common variations
	typeMap := map[string]string{
		"string":  "string",
		"varchar": "string",
		"text":    "string",
		"char":    "string",
		"decimal": "decimal",
		"numeric": "decimal",
		"number":  "decimal",
		"float":   "decimal",
		"double":  "decimal",
		"int":     "integer",
		"bigint":  "integer",
		"integer": "integer",
		"smallint": "integer",
		"tinyint": "integer",
		"date":    "date",
		"timestamp": "timestamp",
		"datetime": "timestamp",
		"boolean": "boolean",
		"bool":    "boolean",
	}
	
	if normalized, ok := typeMap[rawType]; ok {
		return normalized
	}
	return rawType
}
```

Use in `ddlToGraph()`:
```go
columnProps["type"] = normalizeColumnType(column.Type)
```

Use in `toProps()`:
```go
props["type"] = normalizeColumnType(typeKeys[0])
```

### 3. Add Validation Function

**File**: `services/extract/main.go`

Add before `replicateSchema()`:
```go
func validateGraph(nodes []Node, edges []Edge) []string {
	var warnings []string
	
	// Check for orphan columns
	tableMap := make(map[string]bool)
	columnMap := make(map[string]bool)
	hasColumnEdges := make(map[string]bool)
	
	for _, node := range nodes {
		if node.Type == "table" {
			tableMap[node.ID] = true
		} else if node.Type == "column" {
			columnMap[node.ID] = true
		}
	}
	
	for _, edge := range edges {
		if edge.Label == "HAS_COLUMN" && columnMap[edge.TargetID] {
			hasColumnEdges[edge.TargetID] = true
		}
	}
	
	orphanCount := 0
	for colID := range columnMap {
		if !hasColumnEdges[colID] {
			orphanCount++
		}
	}
	
	if orphanCount > 0 {
		warnings = append(warnings, fmt.Sprintf("validation: %d orphan columns found (missing HAS_COLUMN edges)", orphanCount))
	}
	
	// Check for orphan edges
	orphanEdges := 0
	for _, edge := range edges {
		if !tableMap[edge.SourceID] && !columnMap[edge.SourceID] {
			orphanEdges++
		}
		if !tableMap[edge.TargetID] && !columnMap[edge.TargetID] {
			orphanEdges++
		}
	}
	
	if orphanEdges > 0 {
		warnings = append(warnings, fmt.Sprintf("validation: %d orphan edges found", orphanEdges))
	}
	
	return warnings
}
```

Call before replication:
```go
validationWarnings := validateGraph(nodes, edges)
for _, warning := range validationWarnings {
	s.logger.Printf("validation warning: %s", warning)
}
```

### 4. Enhance Column Profiling

**File**: `services/extract/main.go` - `toProps()` method

Improve to extract more metadata:
```go
func (p *columnProfile) toProps() map[string]any {
	props := map[string]any{}
	
	typeKeys := make([]string, 0, len(p.counts))
	for key := range p.counts {
		typeKeys = append(typeKeys, key)
	}
	sort.Strings(typeKeys)
	
	switch len(typeKeys) {
	case 0:
		props["type"] = "unknown"
	case 1:
		props["type"] = normalizeColumnType(typeKeys[0])
	default:
		props["type"] = "mixed"
		props["types"] = typeKeys
	}
	
	nullable := p.nullCount > 0 || (p.totalRows > 0 && p.presentCount < p.totalRows)
	props["nullable"] = nullable
	
	if p.totalRows > 0 {
		props["presence_ratio"] = float64(p.presentCount) / float64(p.totalRows)
		props["null_ratio"] = float64(p.nullCount) / float64(p.totalRows)
		
		// Distinct ratio (approximate)
		distinctEstimate := len(p.counts)
		if p.nullCount > 0 {
			distinctEstimate++ // null is distinct
		}
		props["distinct_ratio"] = float64(distinctEstimate) / float64(p.totalRows)
	}
	
	// Add examples if available
	if len(p.examples) > 0 {
		props["examples"] = p.examples
	}
	
	// Infer not-null constraint
	if p.nullCount == 0 && p.totalRows > 0 {
		props["not_null"] = true
	}
	
	return mapOrNil(props)
}
```

## Testing

After implementing:
1. Run extraction on test data
2. Check for warnings in logs
3. Run `./scripts/check_orphans.sh`
4. Verify quality metrics improve

## Priority

1. **High**: Fix orphan columns (add to normalization)
2. **High**: Add validation function
3. **Medium**: Normalize type names
4. **Medium**: Enhance column profiling
5. **Low**: Extract more DDL metadata

