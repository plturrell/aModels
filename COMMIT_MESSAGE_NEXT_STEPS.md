feat: implement orphan column fix, validation, and type normalization

This commit implements comprehensive improvements to the glean catalog extraction
process to improve data quality and fix orphan column issues.

## Code Improvements

### 1. Orphan Column Fix (normalization.go)
- Added `fixOrphanColumns()` function that automatically creates missing
  HAS_COLUMN edges during normalization
- Uses 3 matching strategies:
  1. Direct prefix match (table.column, schema.table.column)
  2. Cleaned backticks match (handles `table`.`column` format)
  3. Label pattern match (if column ID contains table label)
- Prevents future orphan columns during extraction

### 2. Graph Validation (main.go)
- Added `validateGraph()` function that validates graph integrity before persistence
- Checks for orphan columns (missing HAS_COLUMN edges)
- Checks for orphan edges (pointing to non-existent nodes)
- Logs warnings but doesn't block persistence

### 3. Type Normalization (main.go, ddl.go)
- Added `normalizeColumnType()` function to normalize type names consistently
- Maps variations: STRING/string/varchar → "string"
- Maps variations: DECIMAL/numeric/number → "decimal"
- Maps variations: INTEGER/int/bigint → "integer"
- Applied to both JSON and DDL extraction

## Build Fixes

### 1. Removed OpenAI Dependency
- Removed OpenAI stub (not used in extract service)
- Fixed test dependencies that referenced llms/openai

### 2. Fixed Arrow Version Mismatch
- Updated go-arrow submodule references from v18 to v16
- Fixed internal package imports to match module path
- Arrow Flight functionality properly configured

### 3. Fixed Compilation Errors
- Removed duplicate `normalizeColumnType()` from ddl.go
- Fixed unused variables in normalization.go
- Fixed chains API usage (orchestration is optional)

## Files Modified

- services/extract/normalization.go - Added fixOrphanColumns()
- services/extract/main.go - Added validateGraph(), normalizeColumnType()
- services/extract/ddl.go - Removed duplicate, uses normalizeColumnType()
- services/extract/Dockerfile - Added orchestration dependency, go mod tidy

## Impact

- ✅ Future extractions will automatically fix orphan columns
- ✅ Consistent type names across all extractions
- ✅ Better validation and error detection
- ✅ Cleaner build without unnecessary dependencies

## Testing

After this commit:
- Re-extract SGMI data to verify orphan fixes work
- Check logs for "fixed X orphan columns" messages
- Verify orphan count is reduced on new extractions

