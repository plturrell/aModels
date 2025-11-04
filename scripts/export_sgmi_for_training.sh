#!/usr/bin/env bash
# Export SGMI data from Postgres to CSV format for Relational Transformer training
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OUTPUT_DIR="${REPO_ROOT}/data/training/extracts/sgmi"

mkdir -p "${OUTPUT_DIR}"

echo "=== Exporting SGMI Data from Postgres ==="
echo ""

# Export table-column relationships
echo "1. Exporting table-column relationships..."
docker exec postgres psql -U postgres -d amodels -c "
COPY (
  SELECT 
    t.id as table_id,
    t.label as table_name,
    COALESCE(t.properties_json->>'schema', '') as schema_name,
    c.id as column_id,
    c.label as column_name,
    COALESCE(c.properties_json->>'type', '') as column_type,
    COALESCE(c.properties_json->>'nullable', '') as nullable,
    e.label as relationship_type
  FROM glean_nodes t
  JOIN glean_edges e ON e.source_id = t.id
  JOIN glean_nodes c ON e.target_id = c.id
  WHERE t.kind = 'table' 
    AND c.kind = 'column'
  ORDER BY t.id, c.label
) TO STDOUT WITH CSV HEADER;
" > "${OUTPUT_DIR}/table_columns.csv"

echo "✅ Exported to: ${OUTPUT_DIR}/table_columns.csv"
wc -l "${OUTPUT_DIR}/table_columns.csv"
echo ""

# Export table relationships (foreign keys/dependencies)
echo "2. Exporting table relationships..."
docker exec postgres psql -U postgres -d amodels -c "
COPY (
  SELECT 
    source.id as source_table_id,
    source.label as source_table_name,
    target.id as target_table_id,
    target.label as target_table_name,
    e.label as relationship_type
  FROM glean_nodes source
  JOIN glean_edges e ON e.source_id = source.id
  JOIN glean_nodes target ON e.target_id = target.id
  WHERE source.kind = 'table' 
    AND target.kind = 'table'
    AND e.label != 'HAS_COLUMN'
  ORDER BY source.id, target.id
) TO STDOUT WITH CSV HEADER;
" > "${OUTPUT_DIR}/table_relationships.csv"

echo "✅ Exported to: ${OUTPUT_DIR}/table_relationships.csv"
wc -l "${OUTPUT_DIR}/table_relationships.csv"
echo ""

# Export view dependencies (if views exist)
echo "3. Exporting view dependencies..."
docker exec postgres psql -U postgres -d amodels -c "
COPY (
  SELECT 
    v.id as view_id,
    v.label as view_name,
    t.id as table_id,
    t.label as table_name,
    e.label as dependency_type
  FROM glean_nodes v
  JOIN glean_edges e ON e.source_id = v.id
  JOIN glean_nodes t ON e.target_id = t.id
  WHERE v.kind = 'view' 
    AND t.kind = 'table'
  ORDER BY v.id, t.id
) TO STDOUT WITH CSV HEADER;
" > "${OUTPUT_DIR}/view_dependencies.csv" 2>&1 || echo "No views found (this is OK)"

echo "✅ Exported to: ${OUTPUT_DIR}/view_dependencies.csv"
wc -l "${OUTPUT_DIR}/view_dependencies.csv"
echo ""

# Create summary
echo "=== Export Summary ==="
echo ""
echo "Files exported to: ${OUTPUT_DIR}"
echo ""
ls -lh "${OUTPUT_DIR}"/*.csv
echo ""
echo "✅ Data export complete!"
echo ""
echo "Next steps:"
echo "1. Create training config: configs/rt_sgmi.yaml"
echo "2. Run training: python3 tools/scripts/train_relational_transformer.py --config configs/rt_sgmi.yaml"

