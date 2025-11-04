# Glean Catalog Process Improvements

## Current Issues Identified

### 1. Orphan Columns (8,195 columns)
- **Issue**: Columns exist without `HAS_COLUMN` edges to their parent tables
- **Impact**: Data quality issues, broken relationships
- **Root Cause**: Likely due to:
  - View columns referencing base table columns
  - ID mismatch between columns and tables
  - Missing edges during extraction/normalization

### 2. Missing Properties (Already Fixed)
- **Status**: ✅ Fixed via enrichment script
- **Remaining**: 37 nodes (non-column metadata)

### 3. Data Quality Issues Found
- **Orphan Edges**: ✅ None (good!)
- **Isolated Nodes**: ✅ None (good!)
- **Duplicate Nodes/Edges**: ✅ None (good!)
- **Orphan Columns**: ⚠️ 8,195 (needs fixing)

## Recommended Improvements to Glean Catalog Process

### 1. Fix Orphan Column Relationships

#### A. Add Edge Validation in Normalization

**Location**: `services/extract/normalization.go`

**Current**: Normalization drops edges with missing nodes, but doesn't create missing relationships.

**Improvement**:
```go
// After normalization, add orphan column fix
func fixOrphanColumns(nodes []Node, edges []Edge) ([]Node, []Edge) {
    tableMap := make(map[string]Node)
    for _, node := range nodes {
        if node.Type == "table" {
            tableMap[node.ID] = node
        }
    }
    
    columnMap := make(map[string]Node)
    for _, node := range nodes {
        if node.Type == "column" {
            columnMap[node.ID] = node
        }
    }
    
    // Find columns without table edges
    existingEdges := make(map[string]bool)
    for _, edge := range edges {
        if edge.Label == "HAS_COLUMN" {
            key := fmt.Sprintf("%s->%s", edge.SourceID, edge.TargetID)
            existingEdges[key] = true
        }
    }
    
    // Create missing edges for orphan columns
    for _, col := range columnMap {
        if col.ID == "" {
            continue
        }
        
        // Extract table ID from column ID (format: table.column)
        parts := strings.SplitN(col.ID, ".", 2)
        if len(parts) != 2 {
            continue
        }
        
        tableID := parts[0]
        if table, ok := tableMap[tableID]; ok {
            edgeKey := fmt.Sprintf("%s->%s", table.ID, col.ID)
            if !existingEdges[edgeKey] {
                edges = append(edges, Edge{
                    SourceID: table.ID,
                    TargetID: col.ID,
                    Label:    "HAS_COLUMN",
                })
            }
        }
    }
    
    return nodes, edges
}
```

#### B. Improve DDL Parsing

**Location**: `services/extract/ddl.go` and `scripts/parse_hive_ddl.py`

**Current**: DDL parsing extracts columns but may not create all edges correctly.

**Improvement**:
1. Ensure every column from DDL gets a `HAS_COLUMN` edge
2. Validate edge creation immediately after parsing
3. Add logging for missing relationships

### 2. Enhance Property Extraction

#### A. Improve JSON Column Profiling

**Location**: `services/extract/main.go` - `profileJSONColumns()` and `toProps()`

**Current Issues**:
- Type inference is basic (string, number, boolean, etc.)
- Doesn't distinguish between decimal/int/float
- Doesn't extract size/precision from JSON data
- Missing metadata like constraints, defaults

**Improvements**:
```go
func (p *columnProfile) toProps() map[string]any {
    props := map[string]any{}
    
    // Enhanced type inference
    if len(p.counts) == 1 {
        typeKey := typeKeys[0]
        props["type"] = normalizeType(typeKey) // Normalize to lowercase
        
        // Extract numeric precision if applicable
        if typeKey == "number" && len(p.examples) > 0 {
            if precision := inferPrecision(p.examples); precision != nil {
                props["precision"] = precision
            }
        }
    }
    
    // Add statistics
    props["null_ratio"] = float64(p.nullCount) / float64(p.totalRows)
    props["distinct_ratio"] = float64(len(p.counts)) / float64(p.totalRows)
    
    // Add constraints if detectable
    if p.nullCount == 0 && p.totalRows > 0 {
        props["not_null"] = true
    }
    
    return mapOrNil(props)
}

func normalizeType(t string) string {
    t = strings.ToLower(strings.TrimSpace(t))
    // Normalize common variations
    typeMap := map[string]string{
        "string": "string",
        "varchar": "string",
        "text": "string",
        "decimal": "decimal",
        "numeric": "decimal",
        "number": "decimal",
        "int": "integer",
        "bigint": "integer",
        "integer": "integer",
        "date": "date",
        "timestamp": "timestamp",
        "datetime": "timestamp",
        "boolean": "boolean",
        "bool": "boolean",
    }
    if normalized, ok := typeMap[t]; ok {
        return normalized
    }
    return t
}
```

#### B. Extract More Metadata from DDL

**Location**: `services/extract/ddl.go` - `ddlToGraph()`

**Current**: Extracts basic column properties (type, nullable, size, default, unique, check, references)

**Improvements**:
1. **Extract more table properties**:
   - Storage format (ORC, Parquet, etc.)
   - Location/path
   - Serde properties
   - TBLPROPERTIES

2. **Extract column comments**:
   - Add description/comment to column properties

3. **Extract constraints**:
   - Foreign keys
   - Check constraints
   - Unique constraints

### 3. Add Validation and Quality Checks

#### A. Pre-Persistence Validation

**Location**: `services/extract/main.go` - Before `replicateSchema()` and `SaveGraph()`

**Add**:
```go
func validateGraph(nodes []Node, edges []Edge) []string {
    var warnings []string
    
    // Check for orphan columns
    tableMap := make(map[string]bool)
    for _, node := range nodes {
        if node.Type == "table" {
            tableMap[node.ID] = true
        }
    }
    
    columnMap := make(map[string]bool)
    for _, node := range nodes {
        if node.Type == "column" {
            columnMap[node.ID] = true
        }
    }
    
    edgeMap := make(map[string]bool)
    for _, edge := range edges {
        if edge.Label == "HAS_COLUMN" {
            edgeMap[edge.TargetID] = true
        }
    }
    
    orphanCount := 0
    for colID := range columnMap {
        if !edgeMap[colID] {
            orphanCount++
        }
    }
    
    if orphanCount > 0 {
        warnings = append(warnings, fmt.Sprintf("found %d orphan columns without HAS_COLUMN edges", orphanCount))
    }
    
    return warnings
}
```

#### B. Post-Persistence Validation

**Location**: `services/extract/cmd/extract-validate/main.go`

**Enhance** to check:
- Orphan columns
- Orphan edges
- Missing properties
- Data consistency

### 4. Improve ID Generation and Matching

#### A. Consistent ID Format

**Current**: IDs are generated inconsistently:
- JSON tables: `filename.column`
- DDL tables: `schema.table` or `table`
- Columns: `table.column` or `schema.table.column`

**Improvement**: Standardize ID format:
```go
func normalizeTableID(tableName, schema string) string {
    if schema != "" {
        return fmt.Sprintf("%s.%s", normalizeName(schema), normalizeName(tableName))
    }
    return normalizeName(tableName)
}

func normalizeColumnID(tableID, columnName string) string {
    return fmt.Sprintf("%s.%s", tableID, normalizeName(columnName))
}

func normalizeName(name string) string {
    // Remove backticks, normalize case, trim
    name = strings.Trim(name, "`\"'")
    return strings.TrimSpace(name)
}
```

### 5. Add Orphan Detection and Auto-Fix

#### A. Automatic Orphan Fix

**Location**: `services/extract/normalization.go` - After normalization

**Add function**:
```go
func fixOrphanColumns(nodes []Node, edges []Edge, warnings *[]string) ([]Node, []Edge) {
    // Implementation from section 1.A above
    // Log fixes to warnings
}
```

### 6. Improve Error Handling and Logging

#### A. Better Logging

**Current**: Warnings are logged but not always actionable.

**Improvement**:
- Log orphan column details (ID, expected table)
- Log missing property counts by type
- Log edge creation failures
- Provide actionable warnings

## Implementation Priority

### High Priority (Fix Data Quality)
1. ✅ Fix orphan columns (8,195) - **IMMEDIATE**
2. ✅ Add validation checks
3. ✅ Improve property extraction

### Medium Priority (Improve Quality)
1. Normalize type names (string vs STRING)
2. Extract more metadata from DDL
3. Improve JSON column profiling

### Low Priority (Enhancements)
1. Add column comments
2. Extract storage properties
3. Enhanced constraint detection

## Quick Fixes

### Fix Orphan Columns Now

```sql
-- Fix orphan columns by creating missing HAS_COLUMN edges
INSERT INTO glean_edges (source_id, target_id, label, properties_json, updated_at_utc)
SELECT 
    t.id as source_id,
    c.id as target_id,
    'HAS_COLUMN' as label,
    '{}'::jsonb as properties_json,
    NOW() as updated_at_utc
FROM glean_nodes c
CROSS JOIN glean_nodes t
WHERE c.kind = 'column'
  AND t.kind = 'table'
  AND c.id LIKE t.id || '.%'
  AND NOT EXISTS (
    SELECT 1 FROM glean_edges e 
    WHERE e.source_id = t.id 
      AND e.target_id = c.id 
      AND e.label = 'HAS_COLUMN'
  )
ON CONFLICT (source_id, target_id, label) DO NOTHING;
```

## Testing

After implementing improvements:
1. Run `./scripts/check_orphans.sh` to verify fixes
2. Run `./scripts/run_quality_metrics.sh` to check quality
3. Re-extract a sample to verify improvements

## Next Steps

1. **Immediate**: Fix orphan columns using SQL or code fix
2. **Short-term**: Implement validation and auto-fix in normalization
3. **Medium-term**: Improve property extraction and type normalization
4. **Long-term**: Enhanced metadata extraction and constraint detection

