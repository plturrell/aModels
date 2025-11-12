#!/usr/bin/env bash
# End-to-end test script for SGMI data flow
# Tests complete flow from extraction through storage to training, including AgentFlow and Open Deep Research

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOG_DIR="${REPO_ROOT}/logs/sgmi_e2e_test"
REPORT_DIR="${REPO_ROOT}/reports/sgmi_e2e"

mkdir -p "${LOG_DIR}"
mkdir -p "${REPORT_DIR}"

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/sgmi_e2e_${TIMESTAMP}.log"
REPORT_FILE="${REPORT_DIR}/sgmi_e2e_${TIMESTAMP}.json"

# Configuration
EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL:-http://localhost:19080}"
GRAPH_SERVICE_URL="${GRAPH_SERVICE_URL:-http://localhost:8081}"
TRAINING_SERVICE_URL="${TRAINING_SERVICE_URL:-http://localhost:8080}"
AGENTFLOW_SERVICE_URL="${AGENTFLOW_SERVICE_URL:-http://localhost:9001}"
DEEP_RESEARCH_URL="${DEEP_RESEARCH_URL:-http://localhost:8085}"
POSTGRES_DSN="${POSTGRES_CATALOG_DSN:-postgresql://postgres:postgres@localhost:5432/amodels?sslmode=disable}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-amodels123}"

# Test results
declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging functions
log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[ERROR] $*" | tee -a "${LOG_FILE}" >&2
}

log_success() {
    echo "[SUCCESS] $*" | tee -a "${LOG_FILE}"
}

# Test functions
run_test() {
    local test_name="$1"
    shift
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log "Running test: ${test_name}"
    
    if "$@"; then
        TEST_RESULTS["${test_name}"]="PASS"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        log_success "Test passed: ${test_name}"
        return 0
    else
        TEST_RESULTS["${test_name}"]="FAIL"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        log_error "Test failed: ${test_name}"
        return 1
    fi
}

# Test 1: Verify SGMI source files exist
test_source_files() {
    local data_dir="${REPO_ROOT}/data/training/sgmi"
    local required_files=(
        "${data_dir}/json_with_changes.json"
        "${data_dir}/hive-ddl/sgmisit_all_tables_statement.hql"
        "${data_dir}/hive-ddl/sgmisitetl_all_tables_statement.hql"
        "${data_dir}/hive-ddl/sgmisitstg_all_tables_statement.hql"
        "${data_dir}/hive-ddl/sgmisit_view.hql"
        "${data_dir}/sgmi-controlm/catalyst migration prod 640.xml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "${file}" ]]; then
            log_error "Required file not found: ${file}"
            return 1
        fi
    done
    
    log_success "All SGMI source files found"
    return 0
}

# Test 2: Run SGMI extraction
test_extraction() {
    local extract_script="${REPO_ROOT}/services/extract/scripts/pipelines/run_sgmi_etl_automated.sh"
    local target_url="${EXTRACT_SERVICE_URL}/knowledge-graph"
    
    if [[ ! -f "${extract_script}" ]]; then
        log_error "Extraction script not found: ${extract_script}"
        return 1
    fi
    
    log "Running SGMI extraction..."
    local start_time=$(date +%s)
    
    if bash "${extract_script}" "${target_url}" >> "${LOG_FILE}" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Extraction completed in ${duration}s"
        echo "${duration}" > "${REPORT_DIR}/extraction_time.txt"
        return 0
    else
        log_error "Extraction failed"
        return 1
    fi
}

# Test 3: Verify Postgres data
test_postgres_data() {
    log "Verifying Postgres data..."
    
    python3 <<PYTHON
import psycopg2
import json
import sys
import os

dsn = os.environ.get('POSTGRES_DSN', '${POSTGRES_DSN}')
project_id = 'sgmi-demo'

try:
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    # Count nodes
    cur.execute("SELECT COUNT(*) FROM glean_nodes WHERE properties_json->>'project_id' = %s", (project_id,))
    node_count = cur.fetchone()[0]
    
    # Count edges
    cur.execute("""
        SELECT COUNT(*) FROM glean_edges e
        JOIN glean_nodes n1 ON e.source_id = n1.id
        WHERE n1.properties_json->>'project_id' = %s
    """, (project_id,))
    edge_count = cur.fetchone()[0]
    
    # Check for Control-M jobs
    cur.execute("""
        SELECT COUNT(*) FROM glean_nodes 
        WHERE kind = 'control-m-job' 
        AND properties_json->>'project_id' = %s
    """, (project_id,))
    controlm_count = cur.fetchone()[0]
    
    # Check for tables
    cur.execute("""
        SELECT COUNT(*) FROM glean_nodes 
        WHERE kind = 'table' 
        AND properties_json->>'project_id' = %s
    """, (project_id,))
    table_count = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    result = {
        "nodes": node_count,
        "edges": edge_count,
        "controlm_jobs": controlm_count,
        "tables": table_count,
        "status": "ok" if node_count > 0 and edge_count > 0 else "error"
    }
    
    print(json.dumps(result))
    
    if node_count == 0 or edge_count == 0:
        sys.exit(1)
        
except Exception as e:
    print(json.dumps({"error": str(e), "status": "error"}))
    sys.exit(1)
PYTHON
    
    if [[ $? -eq 0 ]]; then
        log_success "Postgres data verified"
        return 0
    else
        log_error "Postgres data verification failed"
        return 1
    fi
}

# Test 4: Verify Redis data
test_redis_data() {
    log "Verifying Redis data..."
    
    python3 <<PYTHON
import redis
import json
import sys
import os

redis_url = os.environ.get('REDIS_URL', '${REDIS_URL}')
project_id = 'sgmi-demo'

try:
    # Parse Redis URL
    if redis_url.startswith('redis://'):
        redis_url = redis_url.replace('redis://', '')
    
    parts = redis_url.split('/')
    host_port = parts[0].split(':')
    host = host_port[0] if len(host_port) > 0 else 'localhost'
    port = int(host_port[1]) if len(host_port) > 1 else 6379
    db = int(parts[1]) if len(parts) > 1 else 0
    
    r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    # Count schema nodes
    node_keys = r.keys('schema:node:*')
    node_count = len(node_keys)
    
    # Count schema edges
    edge_keys = r.keys('schema:edge:*')
    edge_count = len(edge_keys)
    
    # Check extract entities queue
    entity_count = r.llen('extract:entities')
    
    result = {
        "nodes": node_count,
        "edges": edge_count,
        "entities": entity_count,
        "status": "ok" if node_count > 0 or edge_count > 0 else "error"
    }
    
    print(json.dumps(result))
    
    if node_count == 0 and edge_count == 0:
        sys.exit(1)
        
except Exception as e:
    print(json.dumps({"error": str(e), "status": "error"}))
    sys.exit(1)
PYTHON
    
    if [[ $? -eq 0 ]]; then
        log_success "Redis data verified"
        return 0
    else
        log_error "Redis data verification failed"
        return 1
    fi
}

# Test 5: Verify Neo4j data
test_neo4j_data() {
    log "Verifying Neo4j data..."
    
    python3 <<PYTHON
from neo4j import GraphDatabase
import json
import sys
import os

uri = os.environ.get('NEO4J_URI', '${NEO4J_URI}')
username = os.environ.get('NEO4J_USERNAME', '${NEO4J_USERNAME}')
password = os.environ.get('NEO4J_PASSWORD', '${NEO4J_PASSWORD}')
project_id = 'sgmi-demo'

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    with driver.session() as session:
        # Count nodes
        result = session.run("MATCH (n:Node) RETURN COUNT(n) as count")
        node_count = result.single()["count"]
        
        # Count edges
        result = session.run("MATCH ()-[r:RELATIONSHIP]->() RETURN COUNT(r) as count")
        edge_count = result.single()["count"]
        
        # Count Control-M jobs
        result = session.run("""
            MATCH (n:Node) 
            WHERE n.type = 'control-m-job' 
            RETURN COUNT(n) as count
        """)
        controlm_count = result.single()["count"]
        
        # Count tables
        result = session.run("""
            MATCH (n:Node) 
            WHERE n.type = 'table' 
            RETURN COUNT(n) as count
        """)
        table_count = result.single()["count"]
    
    driver.close()
    
    result = {
        "nodes": node_count,
        "edges": edge_count,
        "controlm_jobs": controlm_count,
        "tables": table_count,
        "status": "ok" if node_count > 0 and edge_count > 0 else "error"
    }
    
    print(json.dumps(result))
    
    if node_count == 0 or edge_count == 0:
        sys.exit(1)
        
except Exception as e:
    print(json.dumps({"error": str(e), "status": "error"}))
    sys.exit(1)
PYTHON
    
    if [[ $? -eq 0 ]]; then
        log_success "Neo4j data verified"
        return 0
    else
        log_error "Neo4j data verification failed"
        return 1
    fi
}

# Test 6: Test training service consumption
test_training_service() {
    log "Testing training service consumption..."
    
    local start_time=$(date +%s)
    
    # Test graph client query
    local response=$(curl -s -X POST "${TRAINING_SERVICE_URL}/pipeline/run" \
        -H "Content-Type: application/json" \
        -d '{
            "project_id": "sgmi-demo",
            "system_id": "sgmi-demo-system",
            "json_tables": [],
            "hive_ddls": [],
            "control_m_files": [],
            "enable_gnn": true
        }' \
        -w "\n%{http_code}" 2>>"${LOG_FILE}")
    
    local http_code=$(echo "${response}" | tail -n1)
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ "${http_code}" == "200" ]] || [[ "${http_code}" == "202" ]]; then
        log_success "Training service responded in ${duration}s"
        echo "${duration}" > "${REPORT_DIR}/training_time.txt"
        return 0
    else
        log_error "Training service returned HTTP ${http_code}"
        return 1
    fi
}

# Test 7: Test AgentFlow integration
test_agentflow() {
    log "Testing AgentFlow integration..."
    
    local start_time=$(date +%s)
    
    # Test flow execution
    local response=$(curl -s -X POST "${AGENTFLOW_SERVICE_URL}/run" \
        -H "Content-Type: application/json" \
        -d '{
            "flow_id": "processes/sgmi_controlm_pipeline",
            "input_value": "Process SGMI data",
            "ensure": true
        }' \
        -w "\n%{http_code}" 2>>"${LOG_FILE}")
    
    local http_code=$(echo "${response}" | tail -n1)
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ "${http_code}" == "200" ]]; then
        log_success "AgentFlow flow executed in ${duration}s"
        echo "${duration}" > "${REPORT_DIR}/agentflow_time.txt"
        return 0
    else
        log_error "AgentFlow returned HTTP ${http_code}"
        return 1
    fi
}

# Test 8: Test Open Deep Research
test_deep_research() {
    log "Testing Open Deep Research..."
    
    local start_time=$(date +%s)
    
    # Test research query
    local response=$(curl -s -X POST "${DEEP_RESEARCH_URL}/research" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "What data elements exist for SGMI?",
            "context": {"project": "sgmi"},
            "tools": ["sparql_query", "catalog_search"]
        }' \
        -w "\n%{http_code}" \
        --max-time 60 2>>"${LOG_FILE}")
    
    local http_code=$(echo "${response}" | tail -n1)
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ "${http_code}" == "200" ]]; then
        log_success "Deep research query completed in ${duration}s"
        echo "${duration}" > "${REPORT_DIR}/deep_research_time.txt"
        return 0
    else
        log_error "Deep research returned HTTP ${http_code}"
        return 1
    fi
}

# Generate report
generate_report() {
    log "Generating test report..."
    
    python3 <<PYTHON
import json
import os
from datetime import datetime

report = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "test_results": {},
    "summary": {
        "total_tests": ${TOTAL_TESTS},
        "passed": ${PASSED_TESTS},
        "failed": ${FAILED_TESTS},
        "success_rate": ${PASSED_TESTS} / ${TOTAL_TESTS} if ${TOTAL_TESTS} > 0 else 0
    },
    "performance": {}
}

# Add test results
$(for test_name in "${!TEST_RESULTS[@]}"; do
    echo "report['test_results']['${test_name}'] = '${TEST_RESULTS[$test_name]}'"
done)

# Add performance metrics
performance_files = {
    "extraction": "${REPORT_DIR}/extraction_time.txt",
    "training": "${REPORT_DIR}/training_time.txt",
    "agentflow": "${REPORT_DIR}/agentflow_time.txt",
    "deep_research": "${REPORT_DIR}/deep_research_time.txt"
}

for key, filepath in performance_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            report["performance"][key] = float(f.read().strip())

with open("${REPORT_FILE}", 'w') as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
PYTHON
}

# Main execution
main() {
    log "Starting SGMI end-to-end test suite"
    log "Configuration:"
    log "  Extract Service: ${EXTRACT_SERVICE_URL}"
    log "  Graph Service: ${GRAPH_SERVICE_URL}"
    log "  Training Service: ${TRAINING_SERVICE_URL}"
    log "  AgentFlow Service: ${AGENTFLOW_SERVICE_URL}"
    log "  Deep Research: ${DEEP_RESEARCH_URL}"
    
    # Run tests
    run_test "source_files" test_source_files
    run_test "extraction" test_extraction
    run_test "postgres_data" test_postgres_data
    run_test "redis_data" test_redis_data
    run_test "neo4j_data" test_neo4j_data
    run_test "training_service" test_training_service
    run_test "agentflow" test_agentflow
    run_test "deep_research" test_deep_research
    
    # Generate report
    generate_report
    
    # Summary
    log ""
    log "========================================="
    log "Test Summary"
    log "========================================="
    log "Total Tests: ${TOTAL_TESTS}"
    log "Passed: ${PASSED_TESTS}"
    log "Failed: ${FAILED_TESTS}"
    log "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    log "Report: ${REPORT_FILE}"
    log "========================================="
    
    if [[ ${FAILED_TESTS} -eq 0 ]]; then
        log_success "All tests passed!"
        exit 0
    else
        log_error "${FAILED_TESTS} test(s) failed"
        exit 1
    fi
}

# Run main
main "$@"

