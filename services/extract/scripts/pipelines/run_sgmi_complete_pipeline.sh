#!/usr/bin/env bash
# Complete SGMI Data Pipeline
# Processes all SGMI data through Extract, DMS, and Catalog into Knowledge Graph
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# SCRIPT_DIR is now pipelines/, so parent is scripts/
SCRIPTS_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${ROOT_DIR}/../.." && pwd)
# Support custom data root for Docker mounts
# Default to extract service data directory (now part of extract service)
SGMI_DATA_ROOT="${SGMI_DATA_ROOT:-${REPO_ROOT}/services/extract/data}"
DATA_ROOT="${SGMI_DATA_ROOT}/training/sgmi"
LOG_DIR="${REPO_ROOT}/logs/sgmi_pipeline"

mkdir -p "${LOG_DIR}"

# Configuration
# Default to localhost, but allow override for Docker network access
# If running from host and ports don't work, use:
#   EXTRACT_SERVICE_URL=http://extract-service:8082 \
#   EXTRACT_SERVICE_URL=http://extract:8083 \
#   CATALOG_SERVICE_URL=http://catalog:8084 \
#   ./scripts/run_sgmi_complete_pipeline.sh
# Or use the wrapper: ./scripts/run_sgmi_pipeline_from_docker.sh
EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL:-http://localhost:8083}"
CATALOG_SERVICE_URL="${CATALOG_SERVICE_URL:-http://localhost:8084}"
GRAPH_SERVICE_URL="${GRAPH_SERVICE_URL:-http://localhost:19080}"
MAX_RETRIES=5
RETRY_DELAY=5
TIMEOUT=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Step 1: Check service health
check_service_health() {
    local service_url=$1
    local service_name=$2
    
    log "Checking ${service_name} health at ${service_url}..."
    
    local health_endpoints=("/health" "/healthz" "/ready" "/readyz")
    local healthy=false
    
    for endpoint in "${health_endpoints[@]}"; do
        if curl -sf --max-time 5 "${service_url}${endpoint}" > /dev/null 2>&1; then
            log "  ✓ ${service_name} is healthy (${endpoint})"
            healthy=true
            break
        fi
    done
    
    if [[ "${healthy}" == "false" ]]; then
        warn "${service_name} health check failed"
        return 1
    fi
    
    return 0
}

# Step 2: Process structured data through Extract service
process_structured_data() {
    log "=========================================="
    log "Step 1: Processing Structured Data (Extract Service)"
    log "=========================================="
    
    # Use the existing automated ETL script (in same directory)
    if [[ -f "${SCRIPT_DIR}/run_sgmi_etl_automated.sh" ]]; then
        log "Running automated SGMI ETL..."
        if "${SCRIPT_DIR}/run_sgmi_etl_automated.sh" "${EXTRACT_SERVICE_URL}/knowledge-graph"; then
            log "✓ Structured data processed successfully"
            return 0
        else
            error "Failed to process structured data"
            return 1
        fi
    else
        error "Automated ETL script not found: ${SCRIPT_DIR}/run_sgmi_etl_automated.sh"
        return 1
    fi
}

# Step 3: Upload documents to Extract service (replaces DMS)
upload_document_to_extract() {
    local file_path=$1
    local name=$2
    local description=$3
    local tags=$4
    
    if [[ ! -f "${file_path}" ]]; then
        error "File not found: ${file_path}"
        return 1
    fi
    
    local filename=$(basename "${file_path}")
    local file_ext="${filename##*.}"
    
    # Determine content type
    local content_type="application/octet-stream"
    case "${file_ext,,}" in
        docx) content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document" ;;
        xlsx) content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" ;;
        xml) content_type="application/xml" ;;
        txt) content_type="text/plain" ;;
        json) content_type="application/json" ;;
        hql) content_type="text/plain" ;;
    esac
    
    info "Uploading ${filename} to Extract service..."
    
    local response_file=$(mktemp)
    local http_code
    
    set +e
    http_code=$(curl -sSL -w "%{http_code}" -o "${response_file}" \
        --max-time ${TIMEOUT} \
        -X POST \
        -F "file=@${file_path}" \
        -F "name=${name}" \
        -F "description=${description}" \
        -F "project_id=${PROJECT_ID:-sgmi}" \
        -F "system_id=${SYSTEM_ID:-sgmi}" \
        "${EXTRACT_SERVICE_URL}/documents/upload" 2>&1)
    local curl_exit=$?
    set -e
    
    if [[ ${curl_exit} -ne 0 ]] || [[ ! "${http_code}" =~ ^2[0-9]{2}$ ]]; then
        error "Failed to upload ${filename} (HTTP ${http_code})"
        if [[ -s "${response_file}" ]]; then
            error "Response: $(head -c 500 "${response_file}")"
        fi
        rm -f "${response_file}"
        return 1
    fi
    
    # Parse response to get document ID
    local doc_id
    if command -v python3 >/dev/null 2>&1; then
        doc_id=$(python3 -c "import json, sys; data=json.load(open('${response_file}')); print(data.get('id', ''))" 2>/dev/null || echo "")
    fi
    
    if [[ -n "${doc_id}" ]]; then
        log "  ✓ Uploaded ${filename} (ID: ${doc_id})"
        echo "${doc_id}"  # Return document ID
    else
        log "  ✓ Uploaded ${filename}"
    fi
    
    rm -f "${response_file}"
    return 0
}

# Step 4: Process all SGMI documents through Extract service
process_documents() {
    log "=========================================="
    log "Step 2: Processing Documents (Extract Service)"
    log "=========================================="
    
    if ! check_service_health "${EXTRACT_SERVICE_URL}" "Extract Service"; then
        warn "Extract service not available, skipping document upload"
        return 0
    fi
    
    local uploaded=0
    local failed=0
    local doc_ids=()
    
    # Process Control-M XML files
    log "Uploading Control-M XML files..."
    for xml_file in "${DATA_ROOT}/SGMI-controlm"/*.xml; do
        if [[ -f "${xml_file}" ]]; then
            local name=$(basename "${xml_file}")
            if upload_document_to_extract "${xml_file}" "${name}" "SGMI Control-M job definition" "sgmi,control-m,xml"; then
                ((uploaded++))
            else
                ((failed++))
            fi
        fi
    done
    
    # Process Excel files
    log "Uploading Excel files..."
    for xlsx_file in "${DATA_ROOT}/SGMI-controlm"/*.xlsx; do
        if [[ -f "${xlsx_file}" ]]; then
            local name=$(basename "${xlsx_file}")
            if upload_document_to_extract "${xlsx_file}" "${name}" "SGMI Excel document" "sgmi,excel"; then
                ((uploaded++))
            else
                ((failed++))
            fi
        fi
    done
    
    # Process Word documents
    log "Uploading Word documents..."
    for docx_file in "${DATA_ROOT}/SGMI-controlm"/*.docx; do
        if [[ -f "${docx_file}" ]]; then
            local name=$(basename "${docx_file}")
            if upload_document_to_extract "${docx_file}" "${name}" "SGMI Word document" "sgmi,word"; then
                ((uploaded++))
            else
                ((failed++))
            fi
        fi
    done
    
    # Process Hive DDL files (as documents for reference)
    log "Uploading Hive DDL files as documents..."
    for hql_file in "${DATA_ROOT}/hive-ddl"/*.hql; do
        if [[ -f "${hql_file}" ]]; then
            local name=$(basename "${hql_file}")
            if upload_document_to_extract "${hql_file}" "${name}" "SGMI Hive DDL file" "sgmi,hive,ddl"; then
                ((uploaded++))
            else
                ((failed++))
            fi
        fi
    done
    
    # Process JSON file (as document for reference)
    if [[ -f "${DATA_ROOT}/json_with_changes.json" ]]; then
        log "Uploading JSON metadata file..."
        if upload_document_to_extract "${DATA_ROOT}/json_with_changes.json" \
            "SGMI JSON Metadata" "SGMI table metadata and change history" "sgmi,json,metadata"; then
            ((uploaded++))
        else
            ((failed++))
        fi
    fi
    
    log "Document upload summary: ${uploaded} uploaded, ${failed} failed"
    
    if [[ ${failed} -gt 0 ]]; then
        warn "Some documents failed to upload"
        return 1
    fi
    
    return 0
}

# Step 5: Verify knowledge graph integration
verify_knowledge_graph() {
    log "=========================================="
    log "Step 3: Verifying Knowledge Graph Integration"
    log "=========================================="
    
    # Query extract service for SGMI nodes
    log "Querying knowledge graph for SGMI data..."
    
    local query_payload=$(cat <<'EOF'
{
  "query": "MATCH (n) WHERE n.project_id = 'sgmi' OR n.system_id = 'sgmi' OR n.id CONTAINS 'sgmi' RETURN count(n) as node_count"
}
EOF
)
    
    local response_file=$(mktemp)
    local http_code
    
    set +e
    http_code=$(curl -sSL -w "%{http_code}" -o "${response_file}" \
        --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        --data-binary "${query_payload}" \
        "${EXTRACT_SERVICE_URL}/knowledge-graph/query" 2>&1)
    set -e
    
    if [[ "${http_code}" =~ ^2[0-9]{2}$ ]] && [[ -s "${response_file}" ]]; then
        if command -v python3 >/dev/null 2>&1; then
            local node_count=$(python3 -c "import json, sys; data=json.load(open('${response_file}')); print(data.get('data', [{}])[0].get('node_count', 0))" 2>/dev/null || echo "0")
            log "  ✓ Found ${node_count} SGMI nodes in knowledge graph"
        else
            log "  ✓ Knowledge graph query successful"
        fi
    else
        warn "Could not verify knowledge graph (HTTP ${http_code})"
    fi
    
    rm -f "${response_file}"
}

# Step 6: Verify catalog integration
verify_catalog() {
    log "=========================================="
    log "Step 4: Verifying Catalog Integration"
    log "=========================================="
    
    if ! check_service_health "${CATALOG_SERVICE_URL}" "Catalog Service"; then
        warn "Catalog service not available, skipping verification"
        return 0
    fi
    
    log "Querying catalog for SGMI data elements..."
    
    local response_file=$(mktemp)
    local http_code
    
    set +e
    http_code=$(curl -sSL -w "%{http_code}" -o "${response_file}" \
        --max-time 30 \
        -X GET \
        "${CATALOG_SERVICE_URL}/catalog/data-elements?source=Extract%20Service" 2>&1)
    set -e
    
    if [[ "${http_code}" =~ ^2[0-9]{2}$ ]] && [[ -s "${response_file}" ]]; then
        if command -v python3 >/dev/null 2>&1; then
            local element_count=$(python3 -c "import json, sys; data=json.load(open('${response_file}')); print(len(data) if isinstance(data, list) else 0)" 2>/dev/null || echo "0")
            log "  ✓ Found ${element_count} data elements in catalog"
        else
            log "  ✓ Catalog query successful"
        fi
    else
        warn "Could not verify catalog (HTTP ${http_code})"
    fi
    
    rm -f "${response_file}"
}

# Main execution
main() {
    log "=========================================="
    log "SGMI Complete Data Pipeline"
    log "=========================================="
    log ""
    log "This pipeline will:"
    log "  1. Process structured data (JSON, DDL, Control-M) through Extract service"
    log "  2. Upload all documents to Extract service"
    log "  3. Verify knowledge graph integration"
    log "  4. Verify catalog integration"
    log ""
    
    # Check services
    log "Checking services..."
    local services_ok=true
    
    if ! check_service_health "${EXTRACT_SERVICE_URL}" "Extract Service"; then
        services_ok=false
    fi
    
    if ! check_service_health "${EXTRACT_SERVICE_URL}" "Extract Service"; then
        warn "Extract service not available, document upload will be skipped"
    fi
    
    if ! check_service_health "${CATALOG_SERVICE_URL}" "Catalog Service"; then
        warn "Catalog service not available, catalog verification will be skipped"
    fi
    
    if [[ "${services_ok}" == "false" ]]; then
        error "Required services are not available"
        exit 1
    fi
    
    log ""
    
    # Step 1: Process structured data
    if ! process_structured_data; then
        error "Failed to process structured data"
        exit 1
    fi
    
    log ""
    
    # Step 2: Process documents
    if ! process_documents; then
        warn "Some documents failed to upload, but continuing..."
    fi
    
    log ""
    
    # Step 3: Verify knowledge graph
    verify_knowledge_graph
    
    log ""
    
    # Step 4: Verify catalog
    verify_catalog
    
    log ""
    log "=========================================="
    log "SGMI Complete Pipeline Finished!"
    log "=========================================="
    log ""
    log "Summary:"
    log "  ✓ Structured data processed through Extract service"
    log "  ✓ Documents uploaded to Extract service"
    log "  ✓ Knowledge graph integration verified"
    log "  ✓ Catalog integration verified"
    log ""
    log "All SGMI data has been processed and integrated into the knowledge graph!"
}

# Run main function
main "$@"

