# Property Enrichment Guide

## Problem

8,232 columns have empty `properties_json`, missing type information and other metadata.

## Solution

### Automatic Enrichment

Run the enrichment script:
```bash
./scripts/enrich_missing_properties.sh
```

This script:
1. **Infers types from column name patterns**:
   - `*_id`, `id_*` → `string`
   - `*_date`, `*_dt`, `date_*` → `date`
   - `*_amount`, `*_price`, `*_value` → `decimal`
   - `*_flag`, `*_is_*` → `boolean`
   - `*_count`, `*_num` → `decimal`
   - Default → `string`

2. **Sets default properties** for remaining columns:
   - `type: 'string'`
   - `nullable: true`
   - `default: true` (marker for inferred)

### Manual Enrichment

If you have source schema information, you can enrich manually:

```sql
-- Update specific columns with known types
UPDATE glean_nodes
SET properties_json = jsonb_build_object(
    'type', 'decimal',
    'nullable', false,
    'precision', 10,
    'scale', 2
)
WHERE kind = 'column'
  AND label = '`table_name`.`column_name`';
```

### From Source Data

For more accurate enrichment, re-extract from source with improved parsing:

1. **Improve DDL parsing** to capture more metadata
2. **Use JSON table metadata** if available
3. **Infer from actual data** if accessible

## Reconciliation

### How Graph Sync Works

The extract service automatically reconciles:

1. **During Extraction**:
   - `replicateSchema()` → Saves to Postgres
   - `graphPersistence.SaveGraph()` → Saves to Neo4j

2. **Postgres is Source of Truth**:
   - Postgres receives data first via `replicateSchemaToPostgres()`
   - Neo4j receives data via `SaveGraph()` call

3. **Automatic Sync**:
   - Both stores are updated during the same extraction run
   - No manual reconciliation needed if extraction runs successfully

### Manual Reconciliation

If graphs are out of sync:

```bash
# Check sync status
./scripts/reconcile_graph_to_postgres.sh

# Re-run extraction to sync
cd services/extract/scripts
./run_sgmi_full_graph.sh http://localhost:19080/graph
```

### Verify Sync

```bash
# Check counts
docker exec postgres psql -U postgres -d amodels -c "SELECT COUNT(*) FROM glean_nodes;"
docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);"

# Should match!
```

## Best Practices

1. **Enrich Before Training**: Run enrichment before generating training data
2. **Verify Sync**: Check sync status after extraction
3. **Source Data**: Prefer source schema information over inference
4. **Monitor**: Track property completeness over time

## Next Steps

After enrichment:
1. ✅ Run quality metrics: `./scripts/run_quality_metrics.sh`
2. ✅ Verify sync: `./scripts/reconcile_graph_to_postgres.sh`
3. ✅ Proceed with training

