#!/usr/bin/env bash
# Fully automated SGMI ETL processing script
# This script handles all steps: service checks, payload building, and submission
# Enhanced with view lineage metadata and validation features from legacy scripts
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT=$(cd "${ROOT_DIR}/../.." && pwd)
# Support custom data root for Docker mounts
# Default to extract service data directory (now part of extract service)
SGMI_DATA_ROOT="${SGMI_DATA_ROOT:-${REPO_ROOT}/services/extract/data}"
DATA_ROOT="${SGMI_DATA_ROOT}/training/sgmi"
LOG_DIR="${REPO_ROOT}/logs/sgmi_pipeline"

mkdir -p "${LOG_DIR}"

# Configuration
EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL:-http://localhost:8083}"
GRAPH_SERVICE_URL="${GRAPH_SERVICE_URL:-http://localhost:19080}"
TARGET_ENDPOINT="${1:-${EXTRACT_SERVICE_URL}/graph}"
# Gitea configuration for Gitea-first extraction process
# These will be passed to payload builders via environment variables
export GITEA_URL="${GITEA_URL:-}"
export GITEA_TOKEN="${GITEA_TOKEN:-}"
export GITEA_OWNER="${GITEA_OWNER:-extract-service}"
export GITEA_REPO_NAME="${GITEA_REPO_NAME:-sgmi-extracted-code}"
export GITEA_BRANCH="${GITEA_BRANCH:-main}"
export GITEA_BASE_PATH="${GITEA_BASE_PATH:-}"
export GITEA_AUTO_CREATE="${GITEA_AUTO_CREATE:-true}"
export GITEA_DESCRIPTION="${GITEA_DESCRIPTION:-SGMI extracted code and documents}"
MAX_RETRIES=5
RETRY_DELAY=5
TIMEOUT=300

# View lineage output configuration
default_view_store="${REPO_ROOT}/agenticAiETH_layer4_AgentFlow/store"
VIEW_REGISTRY_OUT="${SGMI_VIEW_REGISTRY_OUT:-${default_view_store}/sgmi_view_lineage.json}"
VIEW_SUMMARY_OUT="${SGMI_VIEW_SUMMARY_OUT:-$(dirname "${VIEW_REGISTRY_OUT}")/sgmi_view_summary.json}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Step 1: Check if required data files exist
check_data_files() {
    log "Checking SGMI data files..."
    
    local missing=0
    local required_files=(
        "${DATA_ROOT}/json_with_changes.json"
        "${DATA_ROOT}/hive-ddl/sgmisit_all_tables_statement.hql"
        "${DATA_ROOT}/hive-ddl/sgmisitetl_all_tables_statement.hql"
        "${DATA_ROOT}/hive-ddl/sgmisitstg_all_tables_statement.hql"
        "${DATA_ROOT}/hive-ddl/sgmisit_view.hql"
        "${DATA_ROOT}/SGMI-controlm/catalyst migration prod 640.xml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "${file}" ]]; then
            error "Missing required file: ${file}"
            missing=1
        else
            log "  ✓ Found: $(basename "${file}")"
        fi
    done
    
    if [[ ${missing} -ne 0 ]]; then
        error "Required data files are missing. Please ensure all SGMI data files are in ${DATA_ROOT}"
        return 1
    fi
    
    log "All required data files found"
    return 0
}

# Step 2: Check service health
check_service_health() {
    local service_url=$1
    local service_name=$2
    
    log "Checking ${service_name} health at ${service_url}..."
    
    local health_endpoints=("/health" "/healthz" "/ready")
    local healthy=false
    
    for endpoint in "${health_endpoints[@]}"; do
        if curl -sf --max-time 5 "${service_url}${endpoint}" > /dev/null 2>&1; then
            log "  ✓ ${service_name} is healthy (${endpoint})"
            healthy=true
            break
        fi
    done
    
    if [[ "${healthy}" == "false" ]]; then
        warn "${service_name} health check failed. Will attempt to proceed anyway..."
        return 1
    fi
    
    return 0
}

# Step 3: Wait for service to be ready
wait_for_service() {
    local service_url=$1
    local service_name=$2
    local max_attempts=${3:-30}
    local attempt=0
    
    log "Waiting for ${service_name} to be ready..."
    
    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if curl -sf --max-time 5 "${service_url}/health" > /dev/null 2>&1 || \
           curl -sf --max-time 5 "${service_url}/healthz" > /dev/null 2>&1 || \
           curl -sf --max-time 5 "${service_url}/ready" > /dev/null 2>&1; then
            log "  ✓ ${service_name} is ready"
            return 0
        fi
        
        attempt=$((attempt + 1))
        if [[ ${attempt} -lt ${max_attempts} ]]; then
            echo -n "."
            sleep 2
        fi
    done
    
    echo ""
    warn "${service_name} did not become ready after ${max_attempts} attempts"
    return 1
}

# Step 4: Build SGMI payload
build_payload() {
    local tmp_payload=$(mktemp)
    local view_tmp_dir=$(mktemp -d)
    mkdir -p "${default_view_store}"
    
    # Try using generalized pipeline if config exists, otherwise fall back to SGMI-specific
    local config_file="${SCRIPT_DIR}/sgmi-config.yaml"
    if [[ -f "${config_file}" ]] && [[ -f "${SCRIPT_DIR}/code_view_builder.py" ]]; then
        log "Using generalized pipeline with config: ${config_file}"
        # Expand environment variables in config paths
        export SGMI_DATA_ROOT="${SGMI_DATA_ROOT:-${REPO_ROOT}/services/extract/data}"
        if python3 "${SCRIPT_DIR}/code_view_builder.py" "${tmp_payload}" "${config_file}" 2>&1 | tee /tmp/python_output.log; then
            log "Payload built successfully using generalized pipeline"
            echo "${tmp_payload}"
            return 0
        else
            warn "Generalized pipeline failed, falling back to SGMI-specific builder"
        fi
    fi
    
    # Fallback to SGMI-specific builder
    # Set environment variables for sgmi_view_builder.py
    export SGMI_JSON_FILES="${DATA_ROOT}/json_with_changes.json"
    export SGMI_DDL_FILES="${DATA_ROOT}/hive-ddl/sgmisit_all_tables_statement.hql:${DATA_ROOT}/hive-ddl/sgmisitetl_all_tables_statement.hql:${DATA_ROOT}/hive-ddl/sgmisitstg_all_tables_statement.hql:${DATA_ROOT}/hive-ddl/sgmisit_view.hql"
    export SGMI_CONTROLM_FILES="${DATA_ROOT}/SGMI-controlm/catalyst migration prod 640.xml"
    export SGMI_VIEW_TMP_DIR="${view_tmp_dir}"
    export SGMI_VIEW_REGISTRY_OUT="${VIEW_REGISTRY_OUT}"
    export SGMI_VIEW_SUMMARY_OUT="${VIEW_SUMMARY_OUT}"
    
    # Build payload using Python script
    log "Running Python script: ${SCRIPT_DIR}/sgmi_view_builder.py"
    if ! python3 "${SCRIPT_DIR}/sgmi_view_builder.py" "${tmp_payload}" 2>&1 | tee /tmp/python_output.log; then
        error "Failed to build SGMI payload"
        if [ -f /tmp/python_output.log ]; then
            error "Python script error output:"
            cat /tmp/python_output.log >&2
        fi
        rm -f "${tmp_payload}"
        rm -rf "${view_tmp_dir}"
        return 1
    fi
    
    # Verify payload was created and is valid JSON
    if [[ ! -f "${tmp_payload}" ]] || [[ ! -s "${tmp_payload}" ]]; then
        error "Payload file was not created or is empty"
        rm -f "${tmp_payload}"
        rm -rf "${view_tmp_dir}"
        return 1
    fi
    
    # Validate JSON
    if ! python3 -m json.tool "${tmp_payload}" > /dev/null 2>&1; then
        error "Payload file is not valid JSON"
        rm -f "${tmp_payload}"
        rm -rf "${view_tmp_dir}"
        return 1
    fi
    
    # Convert host paths to container paths if needed
    # Check if we're running on host and service is in container
    local needs_path_conversion=false
    if [[ ! -d "/workspace" ]] && docker ps --format "{{.Names}}" | grep -q "extract-service"; then
        needs_path_conversion=true
    fi
    
    if [[ "${needs_path_conversion}" == "true" ]]; then
        log "Converting paths to container format..."
        python3 <<'PYTHON'
import json
import sys

payload_file = sys.argv[1]
with open(payload_file, 'r') as f:
    payload = json.load(f)

def convert_path(path):
    # Convert /home/aModels/... to /workspace/...
    if path.startswith('/home/aModels/data/'):
        return path.replace('/home/aModels/data/', '/workspace/data/')
    elif path.startswith('/home/aModels/'):
        return path.replace('/home/aModels/', '/workspace/')
    # Also handle relative paths that might be absolute in container
    elif not path.startswith('/'):
        # Keep relative paths as-is, they'll be resolved in container
        return path
    return path

if 'json_tables' in payload:
    payload['json_tables'] = [convert_path(p) for p in payload['json_tables']]
if 'hive_ddls' in payload:
    # For DDLs, check if they're file paths or inline statements
    converted_ddls = []
    for ddl in payload['hive_ddls']:
        if isinstance(ddl, str) and (ddl.startswith('/') or ddl.startswith('./')):
            converted_ddls.append(convert_path(ddl))
        else:
            converted_ddls.append(ddl)  # Keep inline DDL statements as-is
    payload['hive_ddls'] = converted_ddls
if 'control_m_files' in payload:
    payload['control_m_files'] = [convert_path(p) for p in payload['control_m_files']]
if 'sql_queries' in payload:
    # SQL queries are typically inline, but check for file paths
    converted_queries = []
    for query in payload['sql_queries']:
        if isinstance(query, str) and (query.startswith('/') or query.startswith('./')):
            converted_queries.append(convert_path(query))
        else:
            converted_queries.append(query)  # Keep inline SQL as-is
    payload['sql_queries'] = converted_queries

with open(payload_file, 'w') as f:
    json.dump(payload, f, indent=2)
PYTHON
        "${tmp_payload}"
    fi
    
    # Log to stderr so it doesn't interfere with return value
    log "Payload built successfully: ${tmp_payload}" >&2
    log "Payload size: $(du -h "${tmp_payload}" | cut -f1)" >&2
    
    # Return ONLY the payload file path to stdout (no other output)
    echo "${tmp_payload}" >&1
}

# Step 5: Submit payload with retries and validation
submit_payload() {
    local payload_file=$1
    local endpoint=$2
    local attempt=0
    local tmp_response=$(mktemp)
    
    log "Submitting SGMI payload to ${endpoint}..."
    
    # Check if we need to use docker exec (if endpoint is localhost but service is in container)
    local use_docker=false
    if [[ "${endpoint}" == *"localhost"* ]] && docker ps --format "{{.Names}}" | grep -q "extract-service"; then
        # Extract service is in container, but we're on host - use docker exec
        use_docker=true
        # Convert endpoint to container-internal URL
        endpoint=$(echo "${endpoint}" | sed 's|http://localhost:[0-9]*|http://localhost:8082|')
        log "Using docker exec to submit payload (container endpoint: ${endpoint})"
    fi
    
    while [[ ${attempt} -lt ${MAX_RETRIES} ]]; do
        attempt=$((attempt + 1))
        
        if [[ ${attempt} -gt 1 ]]; then
            log "Retry attempt ${attempt}/${MAX_RETRIES}..."
            sleep ${RETRY_DELAY}
        fi
        
        local http_status
        local curl_exit
        
        set +e
        if [[ "${use_docker}" == "true" ]]; then
            # Copy payload to container and submit from inside
            docker cp "${payload_file}" extract-service:/tmp/sgmi_payload_submit.json > /dev/null 2>&1
            http_status=$(docker exec extract-service python3 -c "
import json
import requests
import sys

try:
    with open('/tmp/sgmi_payload_submit.json', 'r') as f:
        payload = json.load(f)
    
    r = requests.post('${endpoint}', json=payload, timeout=${TIMEOUT})
    print(f'{r.status_code}')
    print(r.text)
except Exception as e:
    print(f'500')
    print(f'Error: {str(e)}')
" 2>&1 | tee "${tmp_response}" | head -1)
            curl_exit=$?
            # Get the response body (everything after first line)
            tail -n +2 "${tmp_response}" > "${tmp_response}.body" 2>/dev/null || true
            mv "${tmp_response}.body" "${tmp_response}" 2>/dev/null || true
        else
            http_status=$(curl -sSL -w "%{http_code}" --max-time ${TIMEOUT} -X POST \
                -H "Content-Type: application/json" \
                --data-binary "@${payload_file}" \
                -o "${tmp_response}" \
                "${endpoint}" 2>&1)
            curl_exit=$?
        fi
        set -e
        
        if [[ ${curl_exit} -ne 0 ]]; then
            warn "Request failed (attempt ${attempt}/${MAX_RETRIES}): ${http_status}"
            continue
        fi
        
        # Extract HTTP status code
        if [[ "${http_status}" =~ ^[0-9]+$ ]] && [[ "${http_status}" -ge 200 ]] && [[ "${http_status}" -lt 300 ]]; then
            log "✓ Submission successful (HTTP ${http_status})"
            
            # Validate response
            validation_summary=$(
                python3 - "${tmp_response}" "${http_status}" <<'PY'
import json
import pathlib
import sys

resp_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None

try:
    status = int(sys.argv[2])
except (ValueError, IndexError):
    status = 0

result = {
    "http_status": status,
    "nodes": 0,
    "edges": 0,
    "controlm_jobs": 0,
    "tables": 0,
    "validation": "ok",
}

try:
    if status != 200:
        result["validation"] = "http_error"
        raise SystemExit

    if resp_path is None or not resp_path.exists():
        result["validation"] = "missing_response"
        raise SystemExit

    payload = json.loads(resp_path.read_text(encoding="utf-8"))

    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []
    result["nodes"] = len(nodes)
    result["edges"] = len(edges)
    result["controlm_jobs"] = sum(1 for n in nodes if str(n.get("type")).lower() == "control-m-job")
    result["tables"] = sum(1 for n in nodes if str(n.get("type")).lower() == "table")

    if result["nodes"] == 0 or result["edges"] == 0:
        result["validation"] = "empty_graph"
    elif result["controlm_jobs"] == 0:
        result["validation"] = "missing_controlm"
    elif result["tables"] == 0:
        result["validation"] = "missing_tables"
except json.JSONDecodeError:
    result["validation"] = "invalid_json_response"
except SystemExit:
    pass
except Exception as exc:
    result["validation"] = "validation_exception"
    result["error"] = str(exc)

print(json.dumps(result))
PY
            )
            
            # Parse and display results
            python3 <<'PYTHON'
import json
import sys

try:
    with open(sys.argv[1], 'r') as f:
        result = json.load(f)
    
    nodes = result.get('nodes', 0)
    edges = result.get('edges', 0)
    message = result.get('message', '')
    
    print(f"\n✅ SGMI ETL Processing Complete!")
    print(f"   Nodes created: {nodes}")
    print(f"   Edges created: {edges}")
    if message:
        print(f"   Message: {message}")
    
    # Check for warnings or errors
    if 'warnings' in result:
        print(f"\n⚠️  Warnings: {len(result['warnings'])}")
        for warning in result['warnings'][:5]:
            print(f"   - {warning}")
    
    if 'errors' in result and result['errors']:
        print(f"\n❌ Errors: {len(result['errors'])}")
        for error in result['errors'][:5]:
            print(f"   - {error}")
        sys.exit(1)
    
except Exception as e:
    print(f"Response: {sys.argv[1]}")
    print(f"Error parsing response: {e}")
    sys.exit(0)
PYTHON
            "${tmp_response}"
            
            # Log validation summary
            log_file="${LOG_DIR}/sgmi_pipeline_$(date -u +%Y%m%d).log"
            timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
            echo "${timestamp} ${validation_summary}" >> "${log_file}"
            
            # Postgres validation if configured
            status_code=$(printf '%s' "${validation_summary}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["http_status"])')
            validation_state=$(printf '%s' "${validation_summary}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["validation"])')
            
            if [[ "${status_code}" -eq 200 && "${validation_state}" == "ok" && -n "${POSTGRES_CATALOG_DSN:-}" ]]; then
                log "Validating Postgres replication…"
                if ! db_validation=$(
                    cd "${ROOT_DIR}" && go run ./cmd/extract-validate -timeout 5s
                ); then
                    error "Postgres validation failed"
                    echo "${timestamp} ${db_validation}" >> "${log_file}"
                    rm -f "${tmp_response}"
                    return 1
                fi
                echo "${db_validation}"
                echo "${timestamp} ${db_validation}" >> "${log_file}"
                db_status=$(printf '%s' "${db_validation}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
                if [[ "${db_status}" != "ok" ]]; then
                    error "SGMI Postgres validation failed. See ${log_file} for details."
                    rm -f "${tmp_response}"
                    return 1
                fi
            fi
            
            if [[ "${status_code}" -ne 200 || "${validation_state}" != "ok" ]]; then
                error "SGMI submission failed validation. See ${log_file} for details."
                rm -f "${tmp_response}"
                return 1
            fi
            
            rm -f "${tmp_response}"
            return 0
        else
            warn "HTTP ${http_status} response (attempt ${attempt}/${MAX_RETRIES})"
            
            # Show error response
            if [[ -s "${tmp_response}" ]]; then
                error "Response: $(head -c 500 "${tmp_response}")"
            fi
        fi
    done
    
    error "Failed to submit payload after ${MAX_RETRIES} attempts"
    rm -f "${tmp_response}"
    return 1
}

# Main execution
main() {
    log "=========================================="
    log "SGMI ETL Automated Processing"
    log "=========================================="
    log ""
    
    # Step 1: Check data files
    if ! check_data_files; then
        exit 1
    fi
    log ""
    
    # Step 2: Check service health
    local extract_base="${EXTRACT_SERVICE_URL%/*}"
    if ! check_service_health "${extract_base}" "Extract Service"; then
        warn "Service health check failed, but proceeding..."
    fi
    log ""
    
    # Step 3: Wait for service (optional, non-blocking)
    if [[ "${WAIT_FOR_SERVICE:-true}" == "true" ]]; then
        wait_for_service "${extract_base}" "Extract Service" 10 || true
        log ""
    fi
    
    # Step 4: Build payload
    log "Building SGMI payload..."
    local payload_file
    # Use a temp file to capture the return value
    local return_file=$(mktemp)
    build_payload > "${return_file}" 2>&1
    local build_exit=$?
    payload_file=$(cat "${return_file}" | tail -1 | tr -d '\n\r ')
    rm -f "${return_file}"
    
    if [[ ${build_exit} -ne 0 ]] || [[ -z "${payload_file}" ]] || [[ ! -f "${payload_file}" ]]; then
        error "Failed to build payload (exit: ${build_exit}, file: '${payload_file}')"
        exit 1
    fi
    
    log "Payload ready: ${payload_file}"
    log ""
    
    # Step 5: Submit payload
    if ! submit_payload "${payload_file}" "${TARGET_ENDPOINT}"; then
        error "SGMI ETL processing failed"
        rm -f "${payload_file}"
        exit 1
    fi
    log ""
    
    log "=========================================="
    log "SGMI ETL Processing Complete!"
    log "=========================================="
    log "View lineage metadata: ${VIEW_REGISTRY_OUT}"
    log "View summary: ${VIEW_SUMMARY_OUT}"
    log ""
    
    # Cleanup
    rm -f "${payload_file}"
}

# Run main function
main "$@"

