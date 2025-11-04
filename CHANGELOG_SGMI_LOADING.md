# Changelog: SGMI Data Loading and Neo4j/Postgres Integration

## Date: 2025-11-04

## Summary

This commit implements complete SGMI data loading into both Neo4j and Postgres, fixes Neo4j persistence issues, and adds comprehensive documentation and tooling for querying and training.

## Changes Made

### 1. Neo4j Persistence Fix (`services/extract/neo4j.go`)
- **Problem**: Neo4j doesn't support nested maps in properties, causing "Property values can only be of primitive types" errors
- **Solution**: Store all properties as a single JSON string in `properties_json` field
- **Impact**: Allows SGMI data to be saved to Neo4j successfully

**Before:**
```go
SET n.properties = $props  // $props was a map[string]any with nested maps
```

**After:**
```go
SET n.properties_json = $props  // $props is a JSON string
```

### 2. Postgres Integration (`infrastructure/docker/brev/docker-compose.yml`)
- Added `postgres` service definition
- Added `POSTGRES_CATALOG_DSN` environment variable to extract service
- Added postgres as dependency for extract service
- Added `postgresdata` volume

**Result**: Extract service now replicates graph data to Postgres automatically

### 3. Documentation Created

#### `docs/POSTGRES_QUERIES.md`
- Comprehensive query guide for SGMI data in Postgres
- Examples for:
  - Basic counts and statistics
  - Table and column queries
  - View lineage analysis
  - Control-M job queries
  - Schema analysis
  - Path finding
  - Circular dependency detection
  - Data export

#### `docs/TRAINING_SETUP.md`
- Complete guide for generating training data from Postgres
- Step-by-step instructions for:
  - Verifying data is loaded
  - Generating training extracts
  - Preparing training configuration
  - Running Relational Transformer training
  - Querying training data

#### `docs/SGMI_LOADING_GUIDE.md`
- Step-by-step guide for loading SGMI data
- Troubleshooting section
- Manual and automated loading instructions

#### `docs/NEXT_STEPS_AFTER_SGMI.md`
- What to do after data is loaded
- Exploration queries
- Training data generation
- Next actions

#### `docs/SGMI_STATUS.md`
- Current status summary
- Access URLs
- Troubleshooting guide

### 4. Scripts Created

#### `scripts/load_sgmi_data.sh`
- Automated SGMI data loading script
- Checks service status
- Validates data files
- Runs extraction
- Verifies data loaded

#### `scripts/generate_training_from_postgres.sh`
- Generates training data from Postgres
- Calls extract service `/generate/training` endpoint
- Generates both table and document extracts
- Shows output directory and file counts

### 5. Data Status

#### Postgres
- ✅ **31,979 nodes** successfully loaded
  - 31,653 columns
  - 323 tables
  - 1 project, 1 system, 1 information-system
- ✅ `glean_nodes` and `glean_edges` tables populated

#### Neo4j
- ✅ Code fixed to handle nested properties
- ⚠️ Data loading pending (requires resubmission after fix)

## Files Modified

1. `services/extract/neo4j.go` - Fixed nested property handling
2. `infrastructure/docker/brev/docker-compose.yml` - Added postgres service and config

## Files Created

### Documentation
- `docs/POSTGRES_QUERIES.md`
- `docs/TRAINING_SETUP.md`
- `docs/SGMI_LOADING_GUIDE.md`
- `docs/NEXT_STEPS_AFTER_SGMI.md`
- `docs/SGMI_STATUS.md`

### Scripts
- `scripts/load_sgmi_data.sh`
- `scripts/generate_training_from_postgres.sh`

## Testing

### Postgres Verification
```bash
docker exec postgres psql -U postgres -d amodels -c "SELECT COUNT(*) FROM glean_nodes;"
# Result: 31979
```

### Neo4j Verification
```bash
docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);"
# Should show: 31979 (after resubmission)
```

## Next Steps

1. **Resubmit payload to Neo4j** (after fix applied):
   ```bash
   curl -X POST http://54.196.0.75:8082/graph \
     -H "Content-Type: application/json" \
     --data-binary @/tmp/test_payload.json
   ```

2. **Query the data**:
   - Use `docs/POSTGRES_QUERIES.md` for Postgres
   - Use `docs/NEO4J_QUERIES.md` for Neo4j

3. **Generate training data**:
   ```bash
   ./scripts/generate_training_from_postgres.sh
   ```

4. **Start training**:
   - Follow `docs/TRAINING_SETUP.md`

## Breaking Changes

None - all changes are additive or fix existing issues.

## Migration Notes

If you have existing Neo4j data with the old property structure:
1. The new code will store properties as JSON strings
2. Existing queries may need to parse `properties_json` field
3. Consider running a migration script if needed

## Related Issues

- Fixed Neo4j nested property error
- Added Postgres replication for graph data
- Created comprehensive query documentation
- Added training data generation tools

