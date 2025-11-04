#!/usr/bin/env bash
# Enrich missing properties for columns that have empty properties_json
set -euo pipefail

echo "=== Enriching Missing Properties ==="
echo ""

# Check current state
MISSING_COUNT=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes WHERE kind = 'column' AND (properties_json IS NULL OR properties_json = '{}');" 2>/dev/null | tr -d '[:space:]')
echo "Columns with missing properties: $MISSING_COUNT"
echo ""

# Strategy 1: Infer from column name patterns
echo "📊 Strategy 1: Inferring from column name patterns..."
echo ""

# Infer common patterns
ENRICHED=$(docker exec postgres psql -U postgres -d amodels -t -c "
UPDATE glean_nodes
SET properties_json = jsonb_build_object(
    'type', CASE
        WHEN label ILIKE '%_id' OR label ILIKE 'id_%' THEN 'string'
        WHEN label ILIKE '%_date' OR label ILIKE '%_dt' OR label ILIKE 'date_%' THEN 'date'
        WHEN label ILIKE '%_amount' OR label ILIKE '%_price' OR label ILIKE '%_value' THEN 'decimal'
        WHEN label ILIKE '%_flag' OR label ILIKE '%_is_%' THEN 'boolean'
        WHEN label ILIKE '%_count' OR label ILIKE '%_num' THEN 'decimal'
        ELSE 'string'
    END,
    'nullable', true,
    'inferred', true
)
WHERE kind = 'column'
  AND (properties_json IS NULL OR properties_json = '{}'::jsonb)
  AND label IS NOT NULL;
SELECT COUNT(*) FROM glean_nodes WHERE kind = 'column' AND properties_json->>'inferred' = 'true';
" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]')

echo "✅ Applied pattern-based inference"
echo ""

# Check remaining
REMAINING=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes WHERE kind = 'column' AND (properties_json IS NULL OR properties_json = '{}');" 2>/dev/null | tr -d '[:space:]')
echo "Remaining columns with missing properties: $REMAINING"
echo ""

# Strategy 2: Set default for remaining
if [[ "$REMAINING" -gt 0 ]]; then
    echo "📊 Strategy 2: Setting default properties for remaining columns..."
    docker exec postgres psql -U postgres -d amodels << 'SQL'
    UPDATE glean_nodes
    SET properties_json = jsonb_build_object(
        'type', 'string',
        'nullable', true,
        'default', true
    )
    WHERE kind = 'column'
      AND (properties_json IS NULL OR properties_json = '{}');
SQL
    echo "✅ Applied default properties"
    echo ""
fi

# Final count
FINAL_MISSING=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes WHERE kind = 'column' AND (properties_json IS NULL OR properties_json = '{}');" 2>/dev/null | tr -d '[:space:]')
ENRICHED=$((MISSING_COUNT - FINAL_MISSING))

echo "=== Results ==="
echo "  Original missing: $MISSING_COUNT"
echo "  Enriched: $ENRICHED"
echo "  Remaining: $FINAL_MISSING"
echo ""
echo "✅ Property enrichment complete!"
echo ""
echo "Note: To use source data for more accurate enrichment, re-run the extraction"
echo "      with improved source data parsing or schema information."

