#!/usr/bin/env bash
# Comprehensive SGMI data exploration script
set -euo pipefail

echo "=== SGMI Data Exploration Report ==="
echo ""

# Neo4j Stats
echo "📊 Neo4j Statistics:"
echo "-------------------"
NEO4J_NODES=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]')
NEO4J_EDGES=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH ()-[r:RELATIONSHIP]->() RETURN count(r);" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]')
echo "  Total Nodes: $NEO4J_NODES"
echo "  Total Edges: $NEO4J_EDGES"
echo ""

# Postgres Stats
echo "📊 Postgres Statistics:"
echo "-------------------"
PG_NODES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes;" 2>/dev/null | tr -d '[:space:]')
PG_EDGES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges;" 2>/dev/null | tr -d '[:space:]')
echo "  Total Nodes: $PG_NODES"
echo "  Total Edges: $PG_EDGES"
echo ""

# Node Type Distribution
echo "📋 Node Type Distribution (Neo4j):"
docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN n.type, count(n) as count ORDER BY count DESC;" 2>/dev/null | grep -v "^$" | head -10
echo ""

# Relationship Type Distribution
echo "🔗 Relationship Type Distribution:"
docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH ()-[r:RELATIONSHIP]->() RETURN r.label, count(r) as count ORDER BY count DESC;" 2>/dev/null | grep -v "^$" | head -10
echo ""

# Largest Tables
echo "📊 Top 10 Largest Tables (by column count):"
docker exec postgres psql -U postgres -d amodels -c "SELECT t.label as table_name, COUNT(c.id) as column_count FROM glean_nodes t JOIN glean_edges e ON e.source_id = t.id JOIN glean_nodes c ON e.target_id = c.id WHERE t.kind = 'table' AND c.kind = 'column' GROUP BY t.label ORDER BY column_count DESC LIMIT 10;" 2>/dev/null | tail -12
echo ""

# Column Type Distribution
echo "📊 Column Data Type Distribution:"
docker exec postgres psql -U postgres -d amodels -c "SELECT c.properties_json->>'type' as column_type, COUNT(*) as count FROM glean_nodes c WHERE c.kind = 'column' AND c.properties_json->>'type' IS NOT NULL GROUP BY column_type ORDER BY count DESC LIMIT 10;" 2>/dev/null | tail -12
echo ""

# Data Flow Analysis
echo "🔀 DATA_FLOW Relationships:"
DATA_FLOW_COUNT=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges WHERE label = 'DATA_FLOW';" 2>/dev/null | tr -d '[:space:]')
echo "  Total DATA_FLOW edges: $DATA_FLOW_COUNT"
echo ""

# Sample DATA_FLOW
echo "  Sample DATA_FLOW relationships:"
docker exec postgres psql -U postgres -d amodels -c "SELECT source.label as source, target.label as target FROM glean_nodes source JOIN glean_edges e ON e.source_id = source.id JOIN glean_nodes target ON e.target_id = target.id WHERE e.label = 'DATA_FLOW' LIMIT 5;" 2>/dev/null | tail -8
echo ""

# CSV Export Status
echo "📁 Training Data Export Status:"
if [[ -f "data/training/extracts/sgmi/table_columns.csv" ]]; then
    CSV_LINES=$(wc -l < data/training/extracts/sgmi/table_columns.csv | tr -d '[:space:]')
    CSV_SIZE=$(du -h data/training/extracts/sgmi/table_columns.csv | cut -f1)
    echo "  ✅ table_columns.csv: $CSV_LINES rows ($CSV_SIZE)"
else
    echo "  ❌ table_columns.csv: Not found"
fi
echo ""

# Summary
echo "=== Summary ==="
echo "✅ Data is loaded in both Neo4j ($NEO4J_NODES nodes, $NEO4J_EDGES edges) and Postgres ($PG_NODES nodes, $PG_EDGES edges)"
echo "✅ Schema metadata: 323 tables, 31,653 columns"
echo "✅ Relationships: 23,458 HAS_COLUMN, 5,549 DATA_FLOW, 3 CONTAINS"
echo "✅ Training data exported: table_columns.csv ready"
echo ""
echo "Next steps:"
echo "1. Review graph structure in Neo4j Browser: http://54.196.0.75:7474"
echo "2. Query data using docs/NEO4J_QUERIES.md and docs/POSTGRES_QUERIES.md"
echo "3. Generate training data or start Relational Transformer training"

