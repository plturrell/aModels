#!/usr/bin/env bash
# Fix orphan columns by creating missing HAS_COLUMN edges
set -euo pipefail

echo "=== Fixing Orphan Columns ==="
echo ""

# Check current orphan count
BEFORE=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes c WHERE c.kind = 'column' AND NOT EXISTS (SELECT 1 FROM glean_edges e JOIN glean_nodes t ON e.source_id = t.id WHERE e.target_id = c.id AND t.kind = 'table' AND e.label = 'HAS_COLUMN');" 2>/dev/null | tr -d '[:space:]')
echo "Orphan columns before fix: $BEFORE"
echo ""

# Fix orphan columns - handle various ID patterns
echo "Creating missing HAS_COLUMN edges..."
echo "Strategy 1: Direct ID matching..."
FIXED1=$(docker exec postgres psql -U postgres -d amodels -t -c "
INSERT INTO glean_edges (source_id, target_id, label, properties_json, updated_at_utc)
SELECT DISTINCT
    t.id as source_id,
    c.id as target_id,
    'HAS_COLUMN' as label,
    '{}'::jsonb as properties_json,
    NOW() as updated_at_utc
FROM glean_nodes c
CROSS JOIN glean_nodes t
WHERE c.kind = 'column'
  AND t.kind = 'table'
  AND (c.id LIKE t.id || '.%' OR c.id LIKE t.id || '.`%' OR c.id LIKE REPLACE(t.id, '`', '') || '.%')
  AND NOT EXISTS (
    SELECT 1 FROM glean_edges e 
    WHERE e.source_id = t.id 
      AND e.target_id = c.id 
      AND e.label = 'HAS_COLUMN'
  )
ON CONFLICT (source_id, target_id, label) DO NOTHING;
SELECT COUNT(*) FROM glean_edges WHERE label = 'HAS_COLUMN';
" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]' || echo "0")

echo "Strategy 2: Pattern matching with cleaned IDs..."
FIXED2=$(docker exec postgres psql -U postgres -d amodels -t -c "
WITH cleaned_ids AS (
    SELECT 
        id,
        kind,
        REPLACE(REPLACE(REPLACE(id, '`', ''), '``', ''), '.`', '.') as cleaned_id
    FROM glean_nodes
)
INSERT INTO glean_edges (source_id, target_id, label, properties_json, updated_at_utc)
SELECT DISTINCT
    t.id as source_id,
    c.id as target_id,
    'HAS_COLUMN' as label,
    '{}'::jsonb as properties_json,
    NOW() as updated_at_utc
FROM cleaned_ids c
CROSS JOIN cleaned_ids t
JOIN glean_nodes cn ON cn.id = c.id
JOIN glean_nodes tn ON tn.id = t.id
WHERE c.kind = 'column'
  AND t.kind = 'table'
  AND c.cleaned_id LIKE t.cleaned_id || '.%'
  AND NOT EXISTS (
    SELECT 1 FROM glean_edges e 
    WHERE e.source_id = tn.id 
      AND e.target_id = cn.id 
      AND e.label = 'HAS_COLUMN'
  )
ON CONFLICT (source_id, target_id, label) DO NOTHING;
SELECT COUNT(*) FROM glean_edges WHERE label = 'HAS_COLUMN';
" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]' || echo "0")

# Calculate actual fixes
TOTAL_EDGES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges WHERE label = 'HAS_COLUMN';" 2>/dev/null | tr -d '[:space:]')
FIXED=$((TOTAL_EDGES - 23458))  # Original was 23458

echo "✅ Created $FIXED missing edges"
echo ""

# Check remaining
AFTER=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes c WHERE c.kind = 'column' AND NOT EXISTS (SELECT 1 FROM glean_edges e JOIN glean_nodes t ON e.source_id = t.id WHERE e.target_id = c.id AND t.kind = 'table' AND e.label = 'HAS_COLUMN');" 2>/dev/null | tr -d '[:space:]')
echo "Orphan columns after fix: $AFTER"
echo ""

# Summary
if [[ "$AFTER" -eq 0 ]]; then
    echo "✅ All orphan columns fixed!"
else
    echo "⚠️  $AFTER columns still orphaned"
    echo ""
    echo "These may need manual investigation:"
    docker exec postgres psql -U postgres -d amodels -c "SELECT c.id, c.label FROM glean_nodes c WHERE c.kind = 'column' AND NOT EXISTS (SELECT 1 FROM glean_edges e JOIN glean_nodes t ON e.source_id = t.id WHERE e.target_id = c.id AND t.kind = 'table') LIMIT 10;" 2>/dev/null | tail -12
fi

echo ""
echo "=== Update Edge Count ==="
NEW_EDGE_COUNT=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges;" 2>/dev/null | tr -d '[:space:]')
echo "New total edge count: $NEW_EDGE_COUNT"

