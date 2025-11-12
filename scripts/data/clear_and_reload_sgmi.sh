#!/usr/bin/env bash
# Clear Neo4j graph and reload SGMI data to test new execution tracking, 
# data quality, and performance metrics schema
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
EXTRACT_SCRIPT="${REPO_ROOT}/services/extract/scripts/pipelines/run_sgmi_etl_automated.sh"

echo "=== Clear and Reload SGMI Graph ==="
echo ""

# Step 1: Clear the Neo4j graph (but preserve schema nodes)
echo "1. Clearing Neo4j graph data..."
echo "   (Preserving schema nodes: Execution, QualityIssue, PerformanceMetric, Resource, etc.)"

# Clear all Node and RELATIONSHIP nodes/edges (the main graph data)
docker exec neo4j cypher-shell -u neo4j -p amodels123 <<'CYPHER'
// Delete all Node nodes and their relationships
MATCH (n:Node)
DETACH DELETE n;

// Delete all RELATIONSHIP edges
MATCH ()-[r:RELATIONSHIP]->()
DELETE r;

// Also clear any Execution, QualityIssue, PerformanceMetric nodes from previous runs
// (but keep the schema constraints/indexes)
MATCH (e:Execution)
DETACH DELETE e;

MATCH (q:QualityIssue)
DETACH DELETE q;

MATCH (p:PerformanceMetric)
DETACH DELETE p;

MATCH (m:ExecutionMetrics)
DETACH DELETE m;

RETURN "Graph cleared successfully" AS result;
CYPHER

echo "   ✅ Graph cleared"
echo ""

# Step 2: Check current counts
echo "2. Current graph state:"
NODE_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
EDGE_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH ()-[r:RELATIONSHIP]->() RETURN count(r);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
echo "   Nodes: ${NODE_COUNT}"
echo "   Edges: ${EDGE_COUNT}"
echo ""

# Step 3: Verify extract service is running
echo "3. Checking extract service..."
if ! docker ps --format "{{.Names}}" | grep -q "extract"; then
    echo "   ⚠️  Extract service not running. Starting..."
    cd "${REPO_ROOT}/infrastructure/docker/brev" && docker compose up -d extract
    echo "   ⏳ Waiting for extract service to start..."
    sleep 10
else
    echo "   ✅ Extract service is running"
fi
echo ""

# Step 4: Run SGMI ingestion
echo "4. Running SGMI data ingestion..."
echo "   Script: ${EXTRACT_SCRIPT}"
echo ""

if [[ ! -f "${EXTRACT_SCRIPT}" ]]; then
    echo "   ❌ Extraction script not found: ${EXTRACT_SCRIPT}"
    exit 1
fi

# Determine target URL - try extract service directly if graph service not available
if docker ps --format "{{.Names}}" | grep -q "graph-server"; then
    TARGET_URL="${1:-http://graph-server:19080/graph}"
else
    # Use extract service directly
    TARGET_URL="${1:-http://localhost:8081/graph}"
fi
echo "   Target URL: ${TARGET_URL}"
echo ""

# Run the ingestion
cd "${REPO_ROOT}/services/extract/scripts"
if bash "${EXTRACT_SCRIPT}" "${TARGET_URL}"; then
    echo ""
    echo "   ✅ SGMI ingestion completed"
else
    echo ""
    echo "   ❌ SGMI ingestion failed"
    exit 1
fi
echo ""

# Step 5: Wait a moment for persistence to complete
echo "5. Waiting for graph persistence to complete..."
sleep 5
echo ""

# Step 6: Verify new data loaded
echo "6. Verifying data loaded..."
echo ""

# Check main graph nodes
NODE_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
EDGE_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH ()-[r:RELATIONSHIP]->() RETURN count(r);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
echo "   Main Graph:"
echo "     Nodes: ${NODE_COUNT}"
echo "     Edges: ${EDGE_COUNT}"
echo ""

# Check Execution nodes
EXECUTION_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (e:Execution) RETURN count(e);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
EXECUTION_METRICS_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (m:ExecutionMetrics) RETURN count(m);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
echo "   Execution Tracking:"
echo "     Execution nodes: ${EXECUTION_COUNT}"
echo "     ExecutionMetrics nodes: ${EXECUTION_METRICS_COUNT}"
echo ""

# Check QualityIssue nodes
QUALITY_ISSUE_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (q:QualityIssue) RETURN count(q);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
echo "   Data Quality:"
echo "     QualityIssue nodes: ${QUALITY_ISSUE_COUNT}"
echo ""

# Check PerformanceMetric nodes
PERF_METRIC_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (p:PerformanceMetric) RETURN count(p);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
echo "   Performance Metrics:"
echo "     PerformanceMetric nodes: ${PERF_METRIC_COUNT}"
echo ""

# Show sample Execution nodes
echo "7. Sample Execution nodes:"
docker exec neo4j cypher-shell -u neo4j -p amodels123 <<'CYPHER'
MATCH (e:Execution)
RETURN e.execution_type AS type, e.status AS status, e.entity_id AS entity
ORDER BY e.started_at DESC
LIMIT 5;
CYPHER
echo ""

# Show sample QualityIssue nodes
if [[ "${QUALITY_ISSUE_COUNT}" -gt 0 ]]; then
    echo "8. Sample QualityIssue nodes:"
    docker exec neo4j cypher-shell -u neo4j -p amodels123 <<'CYPHER'
    MATCH (q:QualityIssue)
    RETURN q.issue_type AS type, q.severity AS severity, q.description AS description
    ORDER BY q.created_at DESC
    LIMIT 3;
CYPHER
    echo ""
fi

# Show sample PerformanceMetric nodes
if [[ "${PERF_METRIC_COUNT}" -gt 0 ]]; then
    echo "9. Sample PerformanceMetric nodes:"
    docker exec neo4j cypher-shell -u neo4j -p amodels123 <<'CYPHER'
    MATCH (p:PerformanceMetric)
    RETURN p.metric_type AS type, p.value AS value, p.entity_id AS entity
    ORDER BY p.timestamp DESC
    LIMIT 5;
CYPHER
    echo ""
fi

# Summary
echo "=== Summary ==="
echo "✅ Graph cleared and reloaded"
echo "✅ Main graph: ${NODE_COUNT} nodes, ${EDGE_COUNT} edges"
if [[ "${EXECUTION_COUNT}" -gt 0 ]]; then
    echo "✅ Execution tracking: ${EXECUTION_COUNT} executions, ${EXECUTION_METRICS_COUNT} metrics"
else
    echo "⚠️  Execution tracking: No execution nodes found (may need to check extract service logs)"
fi
if [[ "${QUALITY_ISSUE_COUNT}" -gt 0 ]]; then
    echo "✅ Data quality: ${QUALITY_ISSUE_COUNT} issues tracked"
else
    echo "ℹ️  Data quality: No quality issues found (data may be high quality)"
fi
if [[ "${PERF_METRIC_COUNT}" -gt 0 ]]; then
    echo "✅ Performance metrics: ${PERF_METRIC_COUNT} metrics tracked"
else
    echo "⚠️  Performance metrics: No metrics found (may need to check extract service logs)"
fi
echo ""

