#!/usr/bin/env bash
# Universal code-to-knowledge graph pipeline script
# Supports any codebase via project configuration file
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT=$(cd "${ROOT_DIR}/../.." && pwd)

# Configuration
CONFIG_FILE="${1:-}"
EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL:-http://localhost:8083}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/code_to_kg}"

mkdir -p "${LOG_DIR}"

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

# Check if config file is provided
if [[ -z "${CONFIG_FILE}" ]]; then
    error "Usage: $0 <config_file.yaml>"
    error "Example: $0 sgmi-config.yaml"
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    error "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

log "Using config file: ${CONFIG_FILE}"

# Check extract service health
check_service_health() {
    local service_url=$1
    log "Checking Extract Service health at ${service_url}..."
    
    local health_endpoints=("/health" "/healthz" "/ready")
    local healthy=false
    
    for endpoint in "${health_endpoints[@]}"; do
        if curl -sf --max-time 5 "${service_url}${endpoint}" > /dev/null 2>&1; then
            log "  ✓ Extract Service is healthy (${endpoint})"
            healthy=true
            break
        fi
    done
    
    if [[ "${healthy}" == "false" ]]; then
        warn "Extract Service health check failed. Will attempt to proceed anyway..."
        return 1
    fi
    
    return 0
}

# Build payload using Python script
build_payload() {
    local tmp_payload=$(mktemp)
    local view_tmp_dir=$(mktemp -d)
    
    log "Building payload from config..."
    
    # Use generalized code_view_builder.py
    if [[ -f "${SCRIPT_DIR}/code_view_builder.py" ]]; then
        if python3 "${SCRIPT_DIR}/code_view_builder.py" "${tmp_payload}" "${CONFIG_FILE}" 2>&1 | tee /tmp/python_output.log; then
            log "Payload built successfully"
        else
            error "Failed to build payload"
            if [[ -f /tmp/python_output.log ]]; then
                error "Python script error output:"
                cat /tmp/python_output.log >&2
            fi
            rm -f "${tmp_payload}"
            rm -rf "${view_tmp_dir}"
            return 1
        fi
    else
        error "code_view_builder.py not found at ${SCRIPT_DIR}/code_view_builder.py"
        return 1
    fi
    
    # Verify payload
    if [[ ! -f "${tmp_payload}" ]] || [[ ! -s "${tmp_payload}" ]]; then
        error "Payload file was not created or is empty"
        rm -f "${tmp_payload}"
        rm -rf "${view_tmp_dir}"
        return 1
    fi
    
    if ! python3 -m json.tool "${tmp_payload}" > /dev/null 2>&1; then
        error "Payload file is not valid JSON"
        rm -f "${tmp_payload}"
        rm -rf "${view_tmp_dir}"
        return 1
    fi
    
    echo "${tmp_payload}"
}

# Submit payload to extract service
submit_payload() {
    local payload_file=$1
    local endpoint="${EXTRACT_SERVICE_URL}/knowledge-graph"
    
    log "Submitting payload to ${endpoint}..."
    
    local tmp_response=$(mktemp)
    local http_status
    
    set +e
    http_status=$(curl -sSL -w "%{http_code}" --max-time 300 -X POST \
        -H "Content-Type: application/json" \
        --data-binary "@${payload_file}" \
        -o "${tmp_response}" \
        "${endpoint}" 2>&1)
    local curl_exit=$?
    set -e
    
    if [[ ${curl_exit} -ne 0 ]]; then
        error "Request failed: ${http_status}"
        rm -f "${tmp_response}"
        return 1
    fi
    
    # Extract HTTP status code (last 3 characters)
    local status_code="${http_status: -3}"
    
    if [[ "${status_code}" -ge 200 ]] && [[ "${status_code}" -lt 300 ]]; then
        log "✓ Submission successful (HTTP ${status_code})"
        
        # Parse and display results
        if python3 -m json.tool "${tmp_response}" > /dev/null 2>&1; then
            local nodes=$(python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('nodes', 0))" < "${tmp_response}" 2>/dev/null || echo "0")
            local edges=$(python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('edges', 0))" < "${tmp_response}" 2>/dev/null || echo "0")
            
            log "  Nodes created: ${nodes}"
            log "  Edges created: ${edges}"
        fi
        
        rm -f "${tmp_response}"
        return 0
    else
        error "HTTP ${status_code} response"
        if [[ -s "${tmp_response}" ]]; then
            error "Response: $(head -c 500 "${tmp_response}")"
        fi
        rm -f "${tmp_response}"
        return 1
    fi
}

# Main execution
main() {
    log "=========================================="
    log "Code-to-Knowledge Graph Pipeline"
    log "=========================================="
    log ""
    
    # Check service health
    if ! check_service_health "${EXTRACT_SERVICE_URL}"; then
        warn "Service health check failed, but proceeding..."
    fi
    log ""
    
    # Build payload
    log "Building payload from config..."
    local payload_file
    payload_file=$(build_payload)
    
    if [[ -z "${payload_file}" ]] || [[ ! -f "${payload_file}" ]]; then
        error "Failed to build payload"
        exit 1
    fi
    
    log "Payload ready: ${payload_file}"
    log ""
    
    # Submit payload
    if ! submit_payload "${payload_file}"; then
        error "Pipeline processing failed"
        rm -f "${payload_file}"
        exit 1
    fi
    log ""
    
    log "=========================================="
    log "Pipeline Complete!"
    log "=========================================="
    log ""
    
    # Cleanup
    rm -f "${payload_file}"
}

# Run main function
main "$@"

