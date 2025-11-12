#!/usr/bin/env bash
# Run SGMI ingestion from inside the extract container
set -euo pipefail

echo "=== Running SGMI Ingestion from Extract Container ==="
echo ""

# First, clear the graph
echo "1. Clearing Neo4j graph..."
docker exec neo4j cypher-shell -u neo4j -p amodels123 <<'CYPHER'
MATCH (n:Node) DETACH DELETE n;
MATCH ()-[r:RELATIONSHIP]->() DELETE r;
MATCH (e:Execution) DETACH DELETE e;
MATCH (q:QualityIssue) DETACH DELETE q;
MATCH (p:PerformanceMetric) DETACH DELETE p;
MATCH (m:ExecutionMetrics) DETACH DELETE m;
RETURN "Graph cleared" AS result;
CYPHER
echo "   ✅ Graph cleared"
echo ""

# Build the payload using Python from host (where dependencies are available)
echo "2. Building SGMI payload..."
TMP_PAYLOAD=$(mktemp)
REPO_ROOT="/home/aModels"
DATA_ROOT="${REPO_ROOT}/data/training/sgmi"

cd "${REPO_ROOT}/services/extract/scripts"
export SGMI_JSON_FILES="${DATA_ROOT}/json_with_changes.json"
export SGMI_DDL_FILES="${DATA_ROOT}/hive-ddl/sgmisit_all_tables_statement.hql:${DATA_ROOT}/hive-ddl/sgmisitetl_all_tables_statement.hql:${DATA_ROOT}/hive-ddl/sgmisitstg_all_tables_statement.hql:${DATA_ROOT}/hive-ddl/sgmisit_view.hql"
export SGMI_CONTROLM_FILES="${DATA_ROOT}/SGMI-controlm/catalyst migration prod 640.xml"
python3 pipelines/sgmi_view_builder.py "${TMP_PAYLOAD}"

# Convert host paths to container paths
echo "   Converting paths to container format..."
python3 <<PYTHON
import json
import sys

payload_file = "${TMP_PAYLOAD}"
with open(payload_file, 'r') as f:
    payload = json.load(f)

def convert_path(path):
    # Convert /home/aModels/data/... to /workspace/data/...
    if path.startswith('/home/aModels/data/'):
        return path.replace('/home/aModels/data/', '/workspace/data/')
    return path

if 'json_tables' in payload:
    payload['json_tables'] = [convert_path(p) for p in payload['json_tables']]
if 'hive_ddls' in payload:
    payload['hive_ddls'] = [convert_path(p) for p in payload['hive_ddls']]
if 'control_m_files' in payload:
    payload['control_m_files'] = [convert_path(p) for p in payload['control_m_files']]

with open(payload_file, 'w') as f:
    json.dump(payload, f, indent=2)
PYTHON

echo "   ✅ Payload built with container paths"
echo ""

# Copy payload to container and make the request
echo "3. Submitting to extract service..."
docker cp "${TMP_PAYLOAD}" extract-service:/tmp/sgmi_payload.json

# Make the request from inside the container
docker exec extract-service python3 <<'PYTHON'
import json
import requests
import sys

try:
    with open('/tmp/sgmi_payload.json') as f:
        payload = json.load(f)
    
    r = requests.post('http://localhost:8082/knowledge-graph', json=payload, timeout=60)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        resp = r.json()
        nodes = resp.get('nodes', [])
        edges = resp.get('edges', [])
        print(f"Nodes: {len(nodes)}")
        print(f"Edges: {len(edges)}")
        print("✅ Ingestion successful")
        sys.exit(0)
    else:
        print(f"Error: {r.text[:500]}")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
PYTHON

EXIT_CODE=$?
rm -f "${TMP_PAYLOAD}"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "4. Verifying data in Neo4j..."
    sleep 3
    
    NODE_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
    EDGE_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH ()-[r:RELATIONSHIP]->() RETURN count(r);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
    EXECUTION_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (e:Execution) RETURN count(e);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
    QUALITY_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (q:QualityIssue) RETURN count(q);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
    PERF_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (p:PerformanceMetric) RETURN count(p);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
    
    echo "   Main Graph: ${NODE_COUNT} nodes, ${EDGE_COUNT} edges"
    echo "   Execution nodes: ${EXECUTION_COUNT}"
    echo "   QualityIssue nodes: ${QUALITY_COUNT}"
    echo "   PerformanceMetric nodes: ${PERF_COUNT}"
    echo ""
    echo "✅ SGMI data reloaded successfully!"
else
    echo ""
    echo "❌ Ingestion failed"
    exit 1
fi

