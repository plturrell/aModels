#!/usr/bin/env bash
# Comprehensive quality and information metrics report for SGMI extraction
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OUTPUT_FILE="${REPO_ROOT}/logs/quality_metrics_$(date -u +%Y%m%d_%H%M%S).json"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "=== SGMI Extraction Quality & Information Metrics Report ==="
echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# Basic validation
echo "📊 Basic Validation:"
echo "-------------------"
if docker exec postgres psql -U postgres -d amodels -c "SELECT 1" > /dev/null 2>&1; then
    echo "✅ Postgres connection: OK"
else
    echo "❌ Postgres connection: FAILED"
    exit 1
fi

if docker exec neo4j cypher-shell -u neo4j -p amodels123 "RETURN 1" > /dev/null 2>&1; then
    echo "✅ Neo4j connection: OK"
else
    echo "❌ Neo4j connection: FAILED"
    exit 1
fi
echo ""

# Data completeness
echo "📊 Data Completeness:"
echo "-------------------"
NODE_COUNT=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes;" 2>/dev/null | tr -d '[:space:]')
EDGE_COUNT=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges;" 2>/dev/null | tr -d '[:space:]')
echo "  Total Nodes: $NODE_COUNT"
echo "  Total Edges: $EDGE_COUNT"
RATIO=$(python3 -c "print(f'{($NODE_COUNT / $EDGE_COUNT):.2f}')")
echo "  Node-to-Edge Ratio: $RATIO"
echo ""

# Node type distribution
echo "📊 Node Type Distribution:"
docker exec postgres psql -U postgres -d amodels -c "SELECT kind, COUNT(*) as count, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage FROM glean_nodes GROUP BY kind ORDER BY count DESC;" 2>/dev/null | tail -8
echo ""

# Edge type distribution
echo "📊 Edge Type Distribution:"
docker exec postgres psql -U postgres -d amodels -c "SELECT label, COUNT(*) as count, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage FROM glean_edges GROUP BY label ORDER BY count DESC;" 2>/dev/null | tail -6
echo ""

# Data quality checks
echo "📊 Data Quality Checks:"
echo "-------------------"
MISSING_LABELS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes WHERE label IS NULL OR label = '';" 2>/dev/null | tr -d '[:space:]')
MISSING_IDS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes WHERE id IS NULL OR id = '';" 2>/dev/null | tr -d '[:space:]')
MISSING_PROPS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes WHERE properties_json IS NULL OR properties_json = '{}';" 2>/dev/null | tr -d '[:space:]')
ORPHAN_EDGES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges e WHERE NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.source_id) OR NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.target_id);" 2>/dev/null | tr -d '[:space:]')

echo "  Missing Labels: $MISSING_LABELS"
echo "  Missing IDs: $MISSING_IDS"
echo "  Missing Properties: $MISSING_PROPS"
echo "  Orphan Edges: $ORPHAN_EDGES"
echo ""

# Column data type distribution
echo "📊 Column Data Type Distribution:"
docker exec postgres psql -U postgres -d amodels -c "SELECT c.properties_json->>'type' as data_type, COUNT(*) as count, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage FROM glean_nodes c WHERE c.kind = 'column' AND c.properties_json->>'type' IS NOT NULL GROUP BY data_type ORDER BY count DESC LIMIT 10;" 2>/dev/null | tail -13
echo ""

# Table statistics
echo "📊 Table Statistics:"
TOTAL_TABLES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes WHERE kind = 'table';" 2>/dev/null | tr -d '[:space:]')
TABLES_WITH_COLUMNS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(DISTINCT t.id) FROM glean_nodes t JOIN glean_edges e ON e.source_id = t.id JOIN glean_nodes c ON e.target_id = c.id WHERE t.kind = 'table' AND c.kind = 'column' AND e.label = 'HAS_COLUMN';" 2>/dev/null | tr -d '[:space:]')
AVG_COLUMNS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT ROUND(AVG(col_count), 2) FROM (SELECT t.id, COUNT(c.id) as col_count FROM glean_nodes t JOIN glean_edges e ON e.source_id = t.id JOIN glean_nodes c ON e.target_id = c.id WHERE t.kind = 'table' AND c.kind = 'column' AND e.label = 'HAS_COLUMN' GROUP BY t.id) sub;" 2>/dev/null | tr -d '[:space:]')
MAX_COLUMNS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT MAX(col_count) FROM (SELECT t.id, COUNT(c.id) as col_count FROM glean_nodes t JOIN glean_edges e ON e.source_id = t.id JOIN glean_nodes c ON e.target_id = c.id WHERE t.kind = 'table' AND c.kind = 'column' AND e.label = 'HAS_COLUMN' GROUP BY t.id) sub;" 2>/dev/null | tr -d '[:space:]')

echo "  Total Tables: $TOTAL_TABLES"
echo "  Tables with Columns: $TABLES_WITH_COLUMNS"
echo "  Average Columns per Table: $AVG_COLUMNS"
echo "  Max Columns in Single Table: $MAX_COLUMNS"
echo ""

# Graph connectivity
echo "📊 Graph Connectivity:"
ISOLATED_NODES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes n WHERE NOT EXISTS (SELECT 1 FROM glean_edges e WHERE e.source_id = n.id OR e.target_id = n.id);" 2>/dev/null | tr -d '[:space:]')
HIGH_DEGREE_NODES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM (SELECT n.id, COUNT(e.id) as degree FROM glean_nodes n LEFT JOIN glean_edges e ON e.source_id = n.id OR e.target_id = n.id GROUP BY n.id HAVING COUNT(e.id) > 100) sub;" 2>/dev/null | tr -d '[:space:]')

echo "  Isolated Nodes (no edges): $ISOLATED_NODES"
echo "  High-Degree Nodes (>100 edges): $HIGH_DEGREE_NODES"
echo ""

# DATA_FLOW analysis
echo "📊 DATA_FLOW Analysis:"
DATA_FLOW_COUNT=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges WHERE label = 'DATA_FLOW';" 2>/dev/null | tr -d '[:space:]')
SELF_REFERENTIAL=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges e JOIN glean_nodes n1 ON e.source_id = n1.id JOIN glean_nodes n2 ON e.target_id = n2.id WHERE e.label = 'DATA_FLOW' AND n1.label = n2.label;" 2>/dev/null | tr -d '[:space:]')
CROSS_COLUMN=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges e JOIN glean_nodes n1 ON e.source_id = n1.id JOIN glean_nodes n2 ON e.target_id = n2.id WHERE e.label = 'DATA_FLOW' AND n1.label <> n2.label;" 2>/dev/null | tr -d '[:space:]')

echo "  Total DATA_FLOW edges: $DATA_FLOW_COUNT"
echo "  Self-referential flows: $SELF_REFERENTIAL"
echo "  Cross-column flows: $CROSS_COLUMN"
echo ""

# Information metrics (entropy calculation)
echo "📊 Information Metrics:"
echo "-------------------"
COLUMN_TYPES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT c.properties_json->>'type' FROM glean_nodes c WHERE c.kind = 'column' AND c.properties_json->>'type' IS NOT NULL;" 2>/dev/null | tr -d '[:space:]' | sort | uniq -c | sort -rn)

# Calculate entropy
ENTROPY=$(python3 << 'PYTHON'
import sys
from collections import Counter
import math

types = []
for line in sys.stdin:
    if line.strip():
        types.append(line.strip())

if not types:
    print("0.0")
    sys.exit(0)

counts = Counter(types)
total = len(types)
entropy = 0.0

for count in counts.values():
    prob = count / total
    if prob > 0:
        entropy -= prob * math.log2(prob)

print(f"{entropy:.4f}")
PYTHON
<<< "$COLUMN_TYPES")

echo "  Column Type Entropy: $ENTROPY bits"
echo "  (Higher entropy = more diverse data types)"
echo ""

# Data consistency
echo "📊 Data Consistency:"
echo "-------------------"
DUPLICATE_IDS=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) - COUNT(DISTINCT id) FROM glean_nodes;" 2>/dev/null | tr -d '[:space:]')
DUPLICATE_EDGES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) - COUNT(DISTINCT (source_id, target_id, label)) FROM glean_edges;" 2>/dev/null | tr -d '[:space:]')

echo "  Duplicate Node IDs: $DUPLICATE_IDS"
echo "  Duplicate Edges: $DUPLICATE_EDGES"
echo ""

# Neo4j sync status
echo "📊 Neo4j Sync Status:"
echo "-------------------"
NEO4J_NODES=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]')
NEO4J_EDGES=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH ()-[r:RELATIONSHIP]->() RETURN count(r);" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]')

if [[ "$NEO4J_NODES" == "$NODE_COUNT" ]]; then
    echo "✅ Neo4j nodes match Postgres: $NEO4J_NODES"
else
    echo "⚠️  Neo4j nodes ($NEO4J_NODES) != Postgres nodes ($NODE_COUNT)"
fi

if [[ "$NEO4J_EDGES" == "$EDGE_COUNT" ]]; then
    echo "✅ Neo4j edges match Postgres: $NEO4J_EDGES"
else
    echo "⚠️  Neo4j edges ($NEO4J_EDGES) != Postgres edges ($EDGE_COUNT)"
fi
echo ""

# Summary
echo "=== Summary ==="
echo "✅ Data completeness: $NODE_COUNT nodes, $EDGE_COUNT edges"
echo "✅ Data quality: $MISSING_LABELS missing labels, $MISSING_IDS missing IDs"
echo "✅ Graph connectivity: $ISOLATED_NODES isolated nodes"
echo "✅ Information entropy: $ENTROPY bits"
echo "✅ Neo4j sync: Nodes=$NEO4J_NODES, Edges=$NEO4J_EDGES"
echo ""
echo "📄 Full report saved to: ${OUTPUT_FILE}"
echo ""

# Generate JSON report
python3 << 'PYTHON' > "${OUTPUT_FILE}"
import json
import subprocess
import sys
from datetime import datetime

def run_query(query):
    result = subprocess.run(
        ['docker', 'exec', 'postgres', 'psql', '-U', 'postgres', '-d', 'amodels', '-t', '-c', query],
        capture_output=True, text=True, check=False
    )
    return result.stdout.strip()

def run_neo4j_query(query):
    result = subprocess.run(
        ['docker', 'exec', 'neo4j', 'cypher-shell', '-u', 'neo4j', '-p', 'amodels123', query],
        capture_output=True, text=True, check=False
    )
    return result.stdout.strip()

report = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "summary": {},
    "data_completeness": {},
    "node_distribution": {},
    "edge_distribution": {},
    "data_quality": {},
    "column_types": {},
    "table_statistics": {},
    "graph_connectivity": {},
    "data_flow_analysis": {},
    "information_metrics": {},
    "neo4j_sync": {}
}

# Basic counts
report["data_completeness"]["nodes"] = int(run_query("SELECT COUNT(*) FROM glean_nodes;"))
report["data_completeness"]["edges"] = int(run_query("SELECT COUNT(*) FROM glean_edges;"))
report["data_completeness"]["node_to_edge_ratio"] = round(
    report["data_completeness"]["nodes"] / report["data_completeness"]["edges"], 2
)

# Node distribution
node_types = run_query("SELECT kind, COUNT(*) FROM glean_nodes GROUP BY kind;")
for line in node_types.split('\n'):
    if line.strip():
        parts = line.strip().split('|')
        if len(parts) == 2:
            report["node_distribution"][parts[0].strip()] = int(parts[1].strip())

# Edge distribution
edge_types = run_query("SELECT label, COUNT(*) FROM glean_edges GROUP BY label;")
for line in edge_types.split('\n'):
    if line.strip():
        parts = line.strip().split('|')
        if len(parts) == 2:
            report["edge_distribution"][parts[0].strip()] = int(parts[1].strip())

# Data quality
report["data_quality"]["missing_labels"] = int(run_query("SELECT COUNT(*) FROM glean_nodes WHERE label IS NULL OR label = '';"))
report["data_quality"]["missing_ids"] = int(run_query("SELECT COUNT(*) FROM glean_nodes WHERE id IS NULL OR id = '';"))
report["data_quality"]["missing_properties"] = int(run_query("SELECT COUNT(*) FROM glean_nodes WHERE properties_json IS NULL OR properties_json = '{}';"))
report["data_quality"]["orphan_edges"] = int(run_query("SELECT COUNT(*) FROM glean_edges e WHERE NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.source_id) OR NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.target_id);"))

# Neo4j sync
neo4j_nodes = run_neo4j_query("MATCH (n:Node) RETURN count(n);")
neo4j_edges = run_neo4j_query("MATCH ()-[r:RELATIONSHIP]->() RETURN count(r);")
try:
    report["neo4j_sync"]["nodes"] = int(neo4j_nodes.split()[-1])
    report["neo4j_sync"]["edges"] = int(neo4j_edges.split()[-1])
except:
    report["neo4j_sync"]["nodes"] = None
    report["neo4j_sync"]["edges"] = None

print(json.dumps(report, indent=2))
PYTHON

echo "✅ Quality metrics report complete!"

