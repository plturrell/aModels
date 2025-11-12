#!/usr/bin/env bash
# Explore DATA_FLOW relationships in detail
set -euo pipefail

echo "=== DATA_FLOW Relationships Exploration ==="
echo ""

# Total DATA_FLOW count
echo "📊 Total DATA_FLOW Relationships:"
TOTAL=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges WHERE label = 'DATA_FLOW';" 2>/dev/null | tr -d '[:space:]')
echo "  Total: $TOTAL"
echo ""

# Self-referential vs cross-column
echo "📊 DATA_FLOW Analysis:"
SELF_REF=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges e JOIN glean_nodes n1 ON e.source_id = n1.id JOIN glean_nodes n2 ON e.target_id = n2.id WHERE e.label = 'DATA_FLOW' AND n1.id = n2.id;" 2>/dev/null | tr -d '[:space:]')
CROSS_COL=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges e JOIN glean_nodes n1 ON e.source_id = n1.id JOIN glean_nodes n2 ON e.target_id = n2.id WHERE e.label = 'DATA_FLOW' AND n1.id <> n2.id;" 2>/dev/null | tr -d '[:space:]')
echo "  Self-referential (same column): $SELF_REF"
echo "  Cross-column flows: $CROSS_COL"
echo ""

# Cross-table flows
echo "📊 Cross-Table DATA_FLOW Relationships:"
CROSS_TABLE=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(DISTINCT t1.label || ' -> ' || t2.label) FROM glean_nodes t1 JOIN glean_edges e1 ON e1.source_id = t1.id AND e1.label = 'HAS_COLUMN' JOIN glean_nodes c1 ON e1.target_id = c1.id JOIN glean_edges e2 ON e2.source_id = c1.id AND e2.label = 'DATA_FLOW' JOIN glean_nodes c2 ON e2.target_id = c2.id JOIN glean_edges e3 ON e3.target_id = c2.id AND e3.label = 'HAS_COLUMN' JOIN glean_nodes t2 ON e3.source_id = t2.id WHERE t1.kind = 'table' AND t2.kind = 'table' AND t1.id <> t2.id AND c1.id <> c2.id;" 2>/dev/null | tr -d '[:space:]')
echo "  Unique table-to-table flows: $CROSS_TABLE"
echo ""

# Sample cross-table flows
echo "📋 Sample Cross-Table Data Flows:"
docker exec postgres psql -U postgres -d amodels -c "SELECT t1.label as source_table, c1.label as source_column, c2.label as target_column, t2.label as target_table FROM glean_nodes t1 JOIN glean_edges e1 ON e1.source_id = t1.id AND e1.label = 'HAS_COLUMN' JOIN glean_nodes c1 ON e1.target_id = c1.id JOIN glean_edges e2 ON e2.source_id = c1.id AND e2.label = 'DATA_FLOW' JOIN glean_nodes c2 ON e2.target_id = c2.id JOIN glean_edges e3 ON e3.target_id = c2.id AND e3.label = 'HAS_COLUMN' JOIN glean_nodes t2 ON e3.source_id = t2.id WHERE t1.kind = 'table' AND t2.kind = 'table' AND t1.id <> t2.id AND c1.id <> c2.id LIMIT 10;" 2>/dev/null | tail -15
echo ""

# Top columns by DATA_FLOW connectivity
echo "📊 Top Columns by DATA_FLOW Connectivity:"
docker exec postgres psql -U postgres -d amodels -c "SELECT c.label as column, COUNT(DISTINCT e1.id) + COUNT(DISTINCT e2.id) as total_flows FROM glean_nodes c LEFT JOIN glean_edges e1 ON e1.source_id = c.id AND e1.label = 'DATA_FLOW' LEFT JOIN glean_edges e2 ON e2.target_id = c.id AND e2.label = 'DATA_FLOW' WHERE c.kind = 'column' GROUP BY c.id, c.label ORDER BY total_flows DESC LIMIT 15;" 2>/dev/null | tail -18
echo ""

# Tables with most DATA_FLOW activity
echo "📊 Tables with Most DATA_FLOW Activity:"
docker exec postgres psql -U postgres -d amodels -c "SELECT t.label as table_name, COUNT(DISTINCT e.id) as flow_count FROM glean_nodes t JOIN glean_edges e1 ON e1.source_id = t.id AND e1.label = 'HAS_COLUMN' JOIN glean_nodes c ON e1.target_id = c.id JOIN glean_edges e ON (e.source_id = c.id OR e.target_id = c.id) AND e.label = 'DATA_FLOW' WHERE t.kind = 'table' GROUP BY t.id, t.label ORDER BY flow_count DESC LIMIT 15;" 2>/dev/null | tail -18
echo ""

echo "=== Neo4j Browser Access ==="
echo ""
echo "🌐 Access Neo4j Browser at: http://54.196.0.75:7474"
echo "   Username: neo4j"
echo "   Password: amodels123"
echo ""
echo "📝 Copy and paste these queries into Neo4j Browser:"
echo ""
echo "1. Visualize DATA_FLOW relationships:"
echo "   MATCH path = (c1:Node {type: 'column'})-[r:RELATIONSHIP]->(c2:Node {type: 'column'})"
echo "   WHERE r.label = 'DATA_FLOW'"
echo "   RETURN path LIMIT 50"
echo ""
echo "2. Find cross-table flows:"
echo "   MATCH path = (t1:Node {type: 'table'})-[r1:RELATIONSHIP]->(c1:Node {type: 'column'})"
echo "          -[r2:RELATIONSHIP]->(c2:Node {type: 'column'})"
echo "          <-[r3:RELATIONSHIP]-(t2:Node {type: 'table'})"
echo "   WHERE r1.label = 'HAS_COLUMN'"
echo "     AND r2.label = 'DATA_FLOW'"
echo "     AND r3.label = 'HAS_COLUMN'"
echo "     AND t1.label <> t2.label"
echo "   RETURN path LIMIT 30"
echo ""
echo "📚 See docs/DATA_FLOW_EXPLORATION.md for more queries and details"

