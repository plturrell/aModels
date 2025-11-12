#!/usr/bin/env bash
# Comprehensive orphan information check
set -euo pipefail

echo "=== Orphan Information Check ==="
echo ""

# 1. Orphan edges (edges with missing source or target nodes)
echo "1. Checking Orphan Edges:"
echo "-------------------"
ORPHAN_SOURCE=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges e WHERE NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.source_id);" 2>/dev/null | tr -d '[:space:]')
ORPHAN_TARGET=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges e WHERE NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.target_id);" 2>/dev/null | tr -d '[:space:]')

if [[ "$ORPHAN_SOURCE" -eq 0 && "$ORPHAN_TARGET" -eq 0 ]]; then
    echo "  ✅ No orphan edges found"
else
    echo "  ⚠️  Orphan edges with missing source: $ORPHAN_SOURCE"
    echo "  ⚠️  Orphan edges with missing target: $ORPHAN_TARGET"
    
    if [[ "$ORPHAN_SOURCE" -gt 0 ]]; then
        echo ""
        echo "  Sample orphan edges (missing source):"
        docker exec postgres psql -U postgres -d amodels -c "SELECT e.source_id, e.target_id, e.label FROM glean_edges e WHERE NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.source_id) LIMIT 5;" 2>/dev/null | tail -8
    fi
    
    if [[ "$ORPHAN_TARGET" -gt 0 ]]; then
        echo ""
        echo "  Sample orphan edges (missing target):"
        docker exec postgres psql -U postgres -d amodels -c "SELECT e.source_id, e.target_id, e.label FROM glean_edges e WHERE NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.target_id) LIMIT 5;" 2>/dev/null | tail -8
    fi
fi
echo ""

# 2. Isolated nodes (nodes with no edges)
echo "2. Checking Isolated Nodes:"
echo "-------------------"
ISOLATED=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes n WHERE NOT EXISTS (SELECT 1 FROM glean_edges e WHERE e.source_id = n.id OR e.target_id = n.id);" 2>/dev/null | tr -d '[:space:]')

if [[ "$ISOLATED" -eq 0 ]]; then
    echo "  ✅ No isolated nodes found"
else
    echo "  ⚠️  Isolated nodes: $ISOLATED"
    echo ""
    echo "  Isolated nodes by type:"
    docker exec postgres psql -U postgres -d amodels -c "SELECT n.kind, COUNT(*) as count FROM glean_nodes n WHERE NOT EXISTS (SELECT 1 FROM glean_edges e WHERE e.source_id = n.id OR e.target_id = n.id) GROUP BY n.kind ORDER BY count DESC;" 2>/dev/null | tail -8
fi
echo ""

# 3. Tables without columns
echo "3. Checking Tables Without Columns:"
echo "-------------------"
TABLES_NO_COLS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes t WHERE t.kind = 'table' AND NOT EXISTS (SELECT 1 FROM glean_edges e JOIN glean_nodes c ON e.target_id = c.id WHERE e.source_id = t.id AND c.kind = 'column' AND e.label = 'HAS_COLUMN');" 2>/dev/null | tr -d '[:space:]')

if [[ "$TABLES_NO_COLS" -eq 0 ]]; then
    echo "  ✅ All tables have columns"
else
    echo "  ⚠️  Tables without columns: $TABLES_NO_COLS"
    echo ""
    echo "  Sample tables without columns:"
    docker exec postgres psql -U postgres -d amodels -c "SELECT t.label FROM glean_nodes t WHERE t.kind = 'table' AND NOT EXISTS (SELECT 1 FROM glean_edges e JOIN glean_nodes c ON e.target_id = c.id WHERE e.source_id = t.id AND c.kind = 'column') LIMIT 10;" 2>/dev/null | tail -12
fi
echo ""

# 4. Columns without tables
echo "4. Checking Columns Without Tables:"
echo "-------------------"
COLS_NO_TABLE=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes c WHERE c.kind = 'column' AND NOT EXISTS (SELECT 1 FROM glean_edges e JOIN glean_nodes t ON e.source_id = t.id WHERE e.target_id = c.id AND t.kind = 'table' AND e.label = 'HAS_COLUMN');" 2>/dev/null | tr -d '[:space:]')

if [[ "$COLS_NO_TABLE" -eq 0 ]]; then
    echo "  ✅ All columns belong to tables"
else
    echo "  ⚠️  Columns without tables: $COLS_NO_TABLE"
    echo ""
    echo "  Sample orphan columns:"
    docker exec postgres psql -U postgres -d amodels -c "SELECT c.label, c.id FROM glean_nodes c WHERE c.kind = 'column' AND NOT EXISTS (SELECT 1 FROM glean_edges e JOIN glean_nodes t ON e.source_id = t.id WHERE e.target_id = c.id AND t.kind = 'table') LIMIT 10;" 2>/dev/null | tail -12
fi
echo ""

# 5. DATA_FLOW edges without valid column endpoints
echo "5. Checking DATA_FLOW Edge Validity:"
echo "-------------------"
DATA_FLOW_INVALID=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges e JOIN glean_nodes s ON e.source_id = s.id JOIN glean_nodes t ON e.target_id = t.id WHERE e.label = 'DATA_FLOW' AND (s.kind != 'column' OR t.kind != 'column');" 2>/dev/null | tr -d '[:space:]')

if [[ "$DATA_FLOW_INVALID" -eq 0 ]]; then
    echo "  ✅ All DATA_FLOW edges connect columns"
else
    echo "  ⚠️  DATA_FLOW edges with non-column endpoints: $DATA_FLOW_INVALID"
fi
echo ""

# 6. Duplicate nodes
echo "6. Checking Duplicate Nodes:"
echo "-------------------"
DUPLICATES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) - COUNT(DISTINCT id) FROM glean_nodes;" 2>/dev/null | tr -d '[:space:]')

if [[ "$DUPLICATES" -eq 0 ]]; then
    echo "  ✅ No duplicate node IDs"
else
    echo "  ⚠️  Duplicate node IDs: $DUPLICATES"
    echo ""
    echo "  Sample duplicates:"
    docker exec postgres psql -U postgres -d amodels -c "SELECT id, COUNT(*) as count FROM glean_nodes GROUP BY id HAVING COUNT(*) > 1 LIMIT 5;" 2>/dev/null | tail -8
fi
echo ""

# 7. Duplicate edges
echo "7. Checking Duplicate Edges:"
echo "-------------------"
EDGE_DUPLICATES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) - COUNT(DISTINCT (source_id, target_id, label)) FROM glean_edges;" 2>/dev/null | tr -d '[:space:]')

if [[ "$EDGE_DUPLICATES" -eq 0 ]]; then
    echo "  ✅ No duplicate edges"
else
    echo "  ⚠️  Duplicate edges: $EDGE_DUPLICATES"
fi
echo ""

# Summary
echo "=== Summary ==="
TOTAL_ISSUES=$((ORPHAN_SOURCE + ORPHAN_TARGET + ISOLATED + TABLES_NO_COLS + COLS_NO_TABLE + DATA_FLOW_INVALID + DUPLICATES + EDGE_DUPLICATES))
if [[ "$TOTAL_ISSUES" -eq 0 ]]; then
    echo "✅ No orphan information detected!"
    echo "✅ Graph integrity is good"
else
    echo "⚠️  Total issues found: $TOTAL_ISSUES"
    echo ""
    echo "Issues breakdown:"
    echo "  - Orphan edges: $((ORPHAN_SOURCE + ORPHAN_TARGET))"
    echo "  - Isolated nodes: $ISOLATED"
    echo "  - Tables without columns: $TABLES_NO_COLS"
    echo "  - Columns without tables: $COLS_NO_TABLE"
    echo "  - Invalid DATA_FLOW: $DATA_FLOW_INVALID"
    echo "  - Duplicate nodes: $DUPLICATES"
    echo "  - Duplicate edges: $EDGE_DUPLICATES"
fi
echo ""

