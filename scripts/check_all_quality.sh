#!/usr/bin/env bash
# Comprehensive quality check: properties + orphans + sync
set -euo pipefail

echo "=== Comprehensive Quality Check ==="
echo ""

# Run all checks
echo "1. Property Quality:"
./scripts/run_quality_metrics.sh | grep -A 10 "Data Quality"
echo ""

echo "2. Orphan Information:"
./scripts/check_orphans.sh | grep -A 15 "Summary"
echo ""

echo "3. Graph Reconciliation:"
./scripts/reconcile_graph_to_postgres.sh | grep -A 5 "Current State"
echo ""

echo "=== Overall Status ==="
ORPHAN_COLS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes c WHERE c.kind = 'column' AND NOT EXISTS (SELECT 1 FROM glean_edges e JOIN glean_nodes t ON e.source_id = t.id WHERE e.target_id = c.id AND t.kind = 'table' AND e.label = 'HAS_COLUMN');" 2>/dev/null | tr -d '[:space:]')
MISSING_PROPS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes WHERE properties_json IS NULL OR properties_json = '{}'::jsonb;" 2>/dev/null | tr -d '[:space:]')

if [[ "$ORPHAN_COLS" -eq 0 && "$MISSING_PROPS" -eq 0 ]]; then
    echo "✅ All quality checks passed!"
    echo "✅ Ready for training!"
else
    echo "⚠️  Quality issues found:"
    echo "   - Orphan columns: $ORPHAN_COLS"
    echo "   - Missing properties: $MISSING_PROPS"
    echo ""
    echo "See docs/GLEAN_CATALOG_IMPROVEMENTS_SUMMARY.md for fixes"
fi

