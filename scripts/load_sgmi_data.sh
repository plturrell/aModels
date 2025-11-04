#!/usr/bin/env bash
# Complete SGMI Data Loading Script
# This script loads SGMI data into both Neo4j and Postgres

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
COMPOSE_DIR="${REPO_ROOT}/infrastructure/docker/brev"

echo "=== SGMI Data Loading ==="
echo ""

# Step 1: Check if services are running
echo "1. Checking services..."
if ! docker ps --format "{{.Names}}" | grep -q "neo4j"; then
    echo "   ❌ Neo4j not running. Starting..."
    cd "${COMPOSE_DIR}" && docker compose up -d neo4j
    sleep 5
fi

if ! docker ps --format "{{.Names}}" | grep -q "graph-server"; then
    echo "   ❌ Graph service not running. Starting..."
    cd "${COMPOSE_DIR}" && docker compose up -d graph
    echo "   ⏳ Waiting for graph service to start..."
    sleep 10
fi

if ! docker ps --format "{{.Names}}" | grep -q "postgres"; then
    echo "   ❌ Postgres not running. Starting..."
    cd "${COMPOSE_DIR}" && docker compose up -d postgres
    sleep 5
fi

if ! docker ps --format "{{.Names}}" | grep -q "extract-service"; then
    echo "   ❌ Extract service not running. Starting..."
    cd "${COMPOSE_DIR}" && docker compose up -d extract
    sleep 5
fi

echo "   ✅ Services are running"
echo ""

# Step 2: Check if Postgres DSN is configured
echo "2. Checking Postgres configuration..."
if docker exec extract-service env | grep -q "POSTGRES_CATALOG_DSN"; then
    echo "   ✅ Postgres DSN configured"
    POSTGRES_ENABLED=true
else
    echo "   ⚠️  Postgres DSN not configured (Postgres replication will be skipped)"
    echo "   To enable Postgres, add to docker-compose.yml:"
    echo "   POSTGRES_CATALOG_DSN=postgresql://postgres:postgres@postgres:5432/amodels?sslmode=disable"
    POSTGRES_ENABLED=false
fi
echo ""

# Step 3: Check if SGMI data files exist
echo "3. Checking SGMI data files..."
DATA_ROOT="${REPO_ROOT}/data/training/sgmi"
MISSING=0

required_files=(
    "${DATA_ROOT}/json_with_changes.json"
    "${DATA_ROOT}/hive-ddl/sgmisit_all_tables_statement.hql"
    "${DATA_ROOT}/hive-ddl/sgmisitetl_all_tables_statement.hql"
    "${DATA_ROOT}/hive-ddl/sgmisitstg_all_tables_statement.hql"
    "${DATA_ROOT}/hive-ddl/sgmisit_view.hql"
    "${DATA_ROOT}/sgmi-controlm/catalyst migration prod 640.xml"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "   ✅ $(basename "$file")"
    else
        echo "   ❌ Missing: $(basename "$file")"
        MISSING=1
    fi
done

if [[ $MISSING -ne 0 ]]; then
    echo ""
    echo "❌ Missing required SGMI data files. Please add them to ${DATA_ROOT}"
    exit 1
fi
echo ""

# Step 4: Run SGMI extraction
echo "4. Running SGMI extraction..."
EXTRACT_SCRIPT="${REPO_ROOT}/services/extract/scripts/run_sgmi_full_graph.sh"
TARGET_URL="http://graph-server:19080/graph"

if [[ -f "$EXTRACT_SCRIPT" ]]; then
    echo "   Running: ${EXTRACT_SCRIPT} ${TARGET_URL}"
    echo ""
    
    # Run inside extract container or directly
    if docker exec extract-service test -f /workspace/services/extract/scripts/run_sgmi_full_graph.sh 2>/dev/null; then
        docker exec -w /workspace/services/extract/scripts extract-service \
            ./run_sgmi_full_graph.sh http://graph-server:19080/graph
    else
        # Run from host
        cd "${REPO_ROOT}/services/extract/scripts"
        bash run_sgmi_full_graph.sh "${TARGET_URL}"
    fi
else
    echo "   ❌ Extraction script not found: ${EXTRACT_SCRIPT}"
    exit 1
fi

echo ""

# Step 5: Verify data loaded
echo "5. Verifying data loaded..."

echo "   Checking Neo4j..."
NEO4J_COUNT=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);" 2>/dev/null | tail -1 | tr -d '[:space:]' || echo "0")
if [[ "$NEO4J_COUNT" =~ ^[0-9]+$ ]] && [[ "$NEO4J_COUNT" -gt 0 ]]; then
    echo "   ✅ Neo4j: ${NEO4J_COUNT} nodes loaded"
else
    echo "   ⚠️  Neo4j: 0 nodes (may need to check logs)"
fi

if [[ "$POSTGRES_ENABLED" == "true" ]]; then
    echo "   Checking Postgres..."
    POSTGRES_COUNT=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes;" 2>/dev/null | tr -d '[:space:]' || echo "0")
    if [[ "$POSTGRES_COUNT" =~ ^[0-9]+$ ]] && [[ "$POSTGRES_COUNT" -gt 0 ]]; then
        echo "   ✅ Postgres: ${POSTGRES_COUNT} nodes loaded"
    else
        echo "   ⚠️  Postgres: 0 nodes (check if table exists or replication configured)"
    fi
fi

echo ""
echo "=== Complete ==="
echo ""
echo "Next steps:"
echo "1. Explore data in Neo4j Browser: http://54.196.0.75:7474"
echo "2. Run queries from: docs/NEO4J_QUERIES.md"
echo "3. Check Postgres: docker exec postgres psql -U postgres -d amodels -c 'SELECT COUNT(*) FROM glean_nodes;'"

