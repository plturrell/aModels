# Glean Catalog Process - Improvements Summary

## Current Status

### ✅ Good News
- **No orphan edges**: All edges have valid source and target nodes
- **No isolated nodes**: All nodes are connected
- **No duplicates**: No duplicate node IDs or edges
- **Graph sync**: Neo4j and Postgres are in sync
- **Properties enriched**: 8,195 columns enriched with inferred properties

### ⚠️ Issues Found
- **8,195 orphan columns**: Columns without `HAS_COLUMN` edges to tables
- **37 nodes missing properties**: Mostly metadata nodes (project, system, information-system)

## Root Cause: Orphan Columns

### The Problem
8,195 columns exist in the graph but don't have `HAS_COLUMN` edges connecting them to their parent tables.

### Why This Happens

1. **ID Format Mismatch**: 
   - Column IDs: `` `sgmi_scb_product_dat`.`level0` ``
   - Table IDs: `` `sgmi_scb_product_dat` ``
   - Pattern matching fails due to backticks and nested structures

2. **View Columns**:
   - View columns like `vw_sgmi_scb_product_dat.sgmi_scb_product_dat`.`prod_desc`
   - These reference base table columns but may not have direct table relationships

3. **Extraction Process**:
   - Columns are extracted from views/DDLs
   - Edges may not be created if table ID doesn't match exactly
   - Normalization drops edges if source/target nodes don't exist

## Recommended Code Changes

### 1. Add Orphan Column Fix to Normalization (HIGH PRIORITY)

**File**: `services/extract/normalization.go`

Add function after line 100:
```go
// fixOrphanColumns creates missing HAS_COLUMN edges for orphan columns
func fixOrphanColumns(nodes []Node, edges []Edge) ([]Node, []Edge, []string) {
    var warnings []string
    
    // Index tables and columns
    tableMap := make(map[string]Node)
    columnMap := make(map[string]Node)
    for _, node := range nodes {
        if node.Type == "table" {
            tableMap[node.ID] = node
        } else if node.Type == "column" {
            columnMap[node.ID] = node
        }
    }
    
    // Index existing HAS_COLUMN edges
    hasColumnEdges := make(map[string]bool)
    for _, edge := range edges {
        if edge.Label == "HAS_COLUMN" {
            hasColumnEdges[edge.TargetID] = true
        }
    }
    
    fixed := 0
    for colID, col := range columnMap {
        if hasColumnEdges[colID] {
            continue // Already has edge
        }
        
        // Try to find parent table by multiple strategies
        var parentTable *Node
        
        // Strategy 1: Direct prefix match
        parts := strings.Split(colID, ".")
        if len(parts) >= 2 {
            // Try: first part as table ID
            if table, ok := tableMap[parts[0]]; ok {
                parentTable = &table
            }
            
            // Try: all but last part as table ID
            if parentTable == nil && len(parts) > 2 {
                tableID := strings.Join(parts[:len(parts)-1], ".")
                if table, ok := tableMap[tableID]; ok {
                    parentTable = &table
                }
            }
        }
        
        // Strategy 2: Clean backticks and try again
        if parentTable == nil {
            cleanColID := strings.ReplaceAll(colID, "`", "")
            cleanParts := strings.Split(cleanColID, ".")
            if len(cleanParts) >= 2 {
                cleanTableID := cleanParts[0]
                for tid, table := range tableMap {
                    cleanTID := strings.ReplaceAll(tid, "`", "")
                    if cleanTableID == cleanTID {
                        parentTable = &table
                        break
                    }
                }
            }
        }
        
        // Strategy 3: Match by label (if column label matches table label pattern)
        if parentTable == nil {
            colLabel := strings.Trim(col.Label, "`")
            for tid, table := range tableMap {
                tableLabel := strings.Trim(table.Label, "`")
                if strings.Contains(colID, tableLabel) {
                    parentTable = &table
                    break
                }
            }
        }
        
        if parentTable != nil {
            edges = append(edges, Edge{
                SourceID: parentTable.ID,
                TargetID: colID,
                Label:    "HAS_COLUMN",
            })
            fixed++
        }
    }
    
    if fixed > 0 {
        warnings = append(warnings, fmt.Sprintf("fixed %d orphan columns", fixed))
    }
    
    return nodes, edges, warnings
}
```

Call in `normalizeGraph()`:
```go
// After edge normalization
nodes, edges, fixWarnings := fixOrphanColumns(nodes, edges)
result.Warnings = append(result.Warnings, fixWarnings...)
```

### 2. Improve Type Normalization (MEDIUM PRIORITY)

**File**: `services/extract/main.go`

Add function:
```go
func normalizeColumnType(rawType string) string {
    rawType = strings.ToLower(strings.TrimSpace(rawType))
    typeMap := map[string]string{
        "string": "string", "varchar": "string", "text": "string",
        "decimal": "decimal", "numeric": "decimal", "number": "decimal",
        "int": "integer", "bigint": "integer", "integer": "integer",
        "date": "date", "timestamp": "timestamp", "datetime": "timestamp",
        "boolean": "boolean", "bool": "boolean",
    }
    if normalized, ok := typeMap[rawType]; ok {
        return normalized
    }
    return rawType
}
```

Use in:
- `ddlToGraph()`: `columnProps["type"] = normalizeColumnType(column.Type)`
- `toProps()`: `props["type"] = normalizeColumnType(typeKeys[0])`

### 3. Add Pre-Persistence Validation (HIGH PRIORITY)

**File**: `services/extract/main.go`

Add before `replicateSchema()`:
```go
func validateGraph(nodes []Node, edges []Edge) []string {
    var warnings []string
    
    // Check orphan columns
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
        warnings = append(warnings, fmt.Sprintf("validation: %d orphan columns (missing HAS_COLUMN edges)", orphanCount))
    }
    
    return warnings
}
```

### 4. Enhance Column Property Extraction (MEDIUM PRIORITY)

**File**: `services/extract/main.go` - `toProps()`

Enhance to capture:
- More accurate type inference
- Null ratio
- Distinct ratio
- Presence ratio
- Example values

### 5. Improve ID Generation (LOW PRIORITY)

**File**: `services/extract/ddl.go` and `main.go`

Standardize ID format:
- Remove backticks consistently
- Use consistent schema.table.column format
- Normalize names (lowercase, trim)

## Implementation Plan

### Phase 1: Immediate Fixes (Do Now)
1. ✅ Add `fixOrphanColumns()` to normalization
2. ✅ Add validation function
3. ✅ Test with SGMI data

### Phase 2: Quality Improvements (Next)
1. Add type normalization
2. Enhance property extraction
3. Improve error logging

### Phase 3: Long-term Enhancements
1. Standardize ID generation
2. Extract more DDL metadata
3. Add column comments/descriptions

## Testing

After implementing:
1. Run extraction on test data
2. Check logs for warnings
3. Run `./scripts/check_orphans.sh`
4. Run `./scripts/run_quality_metrics.sh`
5. Verify no orphan columns remain

## Files to Modify

1. `services/extract/normalization.go` - Add orphan fix
2. `services/extract/main.go` - Add validation, type normalization
3. `services/extract/ddl.go` - Improve ID generation
4. `services/extract/cmd/extract-validate/main.go` - Enhance validation

## Expected Results

After fixes:
- ✅ 0 orphan columns
- ✅ All columns have HAS_COLUMN edges
- ✅ Consistent type names
- ✅ Better property extraction
- ✅ Validation warnings for future issues

