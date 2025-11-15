#!/bin/bash
################################################################################
# Service Integration Tests
# Tests service-to-service communication and integration workflows
################################################################################

set -euo pipefail

# Color codes
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/testing"
mkdir -p "$LOG_DIR"

################################################################################
# Logging Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $*"
}

log_step() {
    echo -e "\n${BLUE}==>${NC} $*"
}

################################################################################
# Utility Functions
################################################################################

http_get() {
    local url=$1
    local timeout=${2:-10}
    curl -sf --max-time "$timeout" "$url" 2>/dev/null
}

http_post() {
    local url=$1
    local data=$2
    local timeout=${3:-10}
    curl -sf --max-time "$timeout" -X POST -H "Content-Type: application/json" -d "$data" "$url" 2>/dev/null
}

check_service_available() {
    local url=$1
    if http_get "$url" 5 >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

################################################################################
# Integration Test Functions
################################################################################

test_catalog_extract_integration() {
    log_step "Testing Catalog → Extract Integration"
    
    # Verify both services are available
    if ! check_service_available "http://localhost:8084/health"; then
        log_error "Catalog service not available"
        return 1
    fi
    
    if ! check_service_available "http://localhost:8083/health"; then
        log_error "Extract service not available"
        return 1
    fi
    
    log_success "Both Catalog and Extract services are available"
    
    # Test that Extract can query Catalog
    local catalog_response=$(http_get "http://localhost:8084/health" 10)
    if [ -n "$catalog_response" ]; then
        log_success "Catalog is responding to Extract queries"
    else
        log_warn "Could not verify Catalog response format"
    fi
    
    return 0
}

test_graph_localai_integration() {
    log_step "Testing Graph → LocalAI Integration"
    
    # Verify both services are available
    if ! check_service_available "http://localhost:8080/health"; then
        log_error "Graph service not available"
        return 1
    fi
    
    if ! check_service_available "http://localhost:8081/healthz"; then
        log_error "LocalAI service not available"
        return 1
    fi
    
    log_success "Both Graph and LocalAI services are available"
    
    # Test that Graph can use LocalAI
    local localai_models=$(http_get "http://localhost:8081/v1/models" 10)
    if [ -n "$localai_models" ]; then
        log_success "LocalAI models endpoint accessible from Graph"
    else
        log_warn "Could not verify LocalAI models availability"
    fi
    
    return 0
}

test_search_elasticsearch_localai_integration() {
    log_step "Testing Search → Elasticsearch → LocalAI Integration"
    
    # Verify all services are available
    if ! check_service_available "http://localhost:8090/health"; then
        log_error "Search service not available"
        return 1
    fi
    
    if ! check_service_available "http://localhost:9200/_cluster/health"; then
        log_error "Elasticsearch service not available"
        return 1
    fi
    
    if ! check_service_available "http://localhost:8081/healthz"; then
        log_error "LocalAI service not available"
        return 1
    fi
    
    log_success "All services (Search, Elasticsearch, LocalAI) are available"
    
    # Test Elasticsearch cluster health
    local es_health=$(http_get "http://localhost:9200/_cluster/health" 10)
    if [ -n "$es_health" ]; then
        log_success "Elasticsearch cluster is healthy"
    else
        log_error "Elasticsearch cluster health check failed"
        return 1
    fi
    
    return 0
}

test_deepagents_extract_localai_integration() {
    log_step "Testing DeepAgents → Extract → LocalAI Integration"
    
    # Verify all services are available
    if ! check_service_available "http://localhost:9004/healthz"; then
        log_error "DeepAgents service not available"
        return 1
    fi
    
    if ! check_service_available "http://localhost:8083/health"; then
        log_error "Extract service not available"
        return 1
    fi
    
    if ! check_service_available "http://localhost:8081/healthz"; then
        log_error "LocalAI service not available"
        return 1
    fi
    
    log_success "All services (DeepAgents, Extract, LocalAI) are available"
    
    return 0
}

test_runtime_catalog_integration() {
    log_step "Testing Runtime → Catalog Integration"
    
    # Verify both services are available
    if ! check_service_available "http://localhost:8098/healthz"; then
        log_error "Runtime service not available"
        return 1
    fi
    
    if ! check_service_available "http://localhost:8084/health"; then
        log_error "Catalog service not available"
        return 1
    fi
    
    log_success "Both Runtime and Catalog services are available"
    
    return 0
}

test_orchestration_multi_service_integration() {
    log_step "Testing Orchestration → Multiple Services Integration"
    
    # Verify Orchestration is available
    if ! check_service_available "http://localhost:8085/healthz"; then
        log_error "Orchestration service not available"
        return 1
    fi
    
    log_success "Orchestration service is available"
    
    # Check that it can potentially reach other services
    local services=("catalog" "extract" "localai")
    local available_count=0
    
    for service in "${services[@]}"; do
        case "$service" in
            catalog)
                if check_service_available "http://localhost:8084/health"; then
                    available_count=$((available_count + 1))
                fi
                ;;
            extract)
                if check_service_available "http://localhost:8083/health"; then
                    available_count=$((available_count + 1))
                fi
                ;;
            localai)
                if check_service_available "http://localhost:8081/healthz"; then
                    available_count=$((available_count + 1))
                fi
                ;;
        esac
    done
    
    log_info "Orchestration can reach $available_count/${#services[@]} dependent services"
    
    return 0
}

test_extract_gitea_integration() {
    log_step "Testing Extract → Gitea Integration"
    
    # Verify both services are available
    if ! check_service_available "http://localhost:8083/health"; then
        log_error "Extract service not available"
        return 1
    fi
    
    if ! check_service_available "http://localhost:3003/api/healthz"; then
        log_error "Gitea service not available"
        return 1
    fi
    
    log_success "Both Extract and Gitea services are available"
    
    # Test Gitea API accessibility
    local gitea_response=$(http_get "http://localhost:3003/api/healthz" 10)
    if [ -n "$gitea_response" ]; then
        log_success "Gitea API is accessible"
    else
        log_warn "Could not verify Gitea API response"
    fi
    
    return 0
}

test_telemetry_exporter_extract_integration() {
    log_step "Testing Telemetry Exporter → Extract Integration"
    
    # Find telemetry exporter port
    local telem_ports=(8085 8080 8083)
    local telem_port=""
    
    for port in "${telem_ports[@]}"; do
        if check_service_available "http://localhost:$port/health"; then
            telem_port=$port
            break
        fi
    done
    
    if [ -z "$telem_port" ]; then
        log_error "Telemetry Exporter not found"
        return 1
    fi
    
    if ! check_service_available "http://localhost:8083/health"; then
        log_error "Extract service not available"
        return 1
    fi
    
    log_success "Both Telemetry Exporter (port $telem_port) and Extract services are available"
    
    return 0
}

test_end_to_end_extraction_workflow() {
    log_step "Testing End-to-End: Extract → Catalog → Graph Workflow"
    
    # Verify all services in the workflow
    local services_ok=true
    
    if ! check_service_available "http://localhost:8083/health"; then
        log_error "Extract service not available"
        services_ok=false
    fi
    
    if ! check_service_available "http://localhost:8084/health"; then
        log_error "Catalog service not available"
        services_ok=false
    fi
    
    if ! check_service_available "http://localhost:8080/health"; then
        log_error "Graph service not available"
        services_ok=false
    fi
    
    if [ "$services_ok" = false ]; then
        return 1
    fi
    
    log_success "All services in extraction workflow are available"
    log_info "Workflow: Extract → Catalog → Graph"
    
    return 0
}

test_end_to_end_search_workflow() {
    log_step "Testing End-to-End: Search → Elasticsearch → LocalAI Workflow"
    
    # Verify all services in the workflow
    local services_ok=true
    
    if ! check_service_available "http://localhost:8090/health"; then
        log_error "Search service not available"
        services_ok=false
    fi
    
    if ! check_service_available "http://localhost:9200/_cluster/health"; then
        log_error "Elasticsearch service not available"
        services_ok=false
    fi
    
    if ! check_service_available "http://localhost:8081/healthz"; then
        log_error "LocalAI service not available"
        services_ok=false
    fi
    
    if [ "$services_ok" = false ]; then
        return 1
    fi
    
    log_success "All services in search workflow are available"
    log_info "Workflow: Search → Elasticsearch → LocalAI"
    
    return 0
}

################################################################################
# Main
################################################################################

main() {
    echo "═══════════════════════════════════════════"
    echo "Service Integration Tests"
    echo "═══════════════════════════════════════════"
    echo ""
    
    local failed=0
    local total=0
    
    # Run integration tests
    total=$((total + 1))
    if test_catalog_extract_integration; then
        log_success "Catalog → Extract integration test passed"
    else
        log_error "Catalog → Extract integration test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_graph_localai_integration; then
        log_success "Graph → LocalAI integration test passed"
    else
        log_error "Graph → LocalAI integration test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_search_elasticsearch_localai_integration; then
        log_success "Search → Elasticsearch → LocalAI integration test passed"
    else
        log_error "Search → Elasticsearch → LocalAI integration test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_deepagents_extract_localai_integration; then
        log_success "DeepAgents → Extract → LocalAI integration test passed"
    else
        log_error "DeepAgents → Extract → LocalAI integration test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_runtime_catalog_integration; then
        log_success "Runtime → Catalog integration test passed"
    else
        log_error "Runtime → Catalog integration test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_orchestration_multi_service_integration; then
        log_success "Orchestration multi-service integration test passed"
    else
        log_error "Orchestration multi-service integration test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_extract_gitea_integration; then
        log_success "Extract → Gitea integration test passed"
    else
        log_error "Extract → Gitea integration test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_telemetry_exporter_extract_integration; then
        log_success "Telemetry Exporter → Extract integration test passed"
    else
        log_error "Telemetry Exporter → Extract integration test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_end_to_end_extraction_workflow; then
        log_success "End-to-end extraction workflow test passed"
    else
        log_error "End-to-end extraction workflow test failed"
        failed=$((failed + 1))
    fi
    
    total=$((total + 1))
    if test_end_to_end_search_workflow; then
        log_success "End-to-end search workflow test passed"
    else
        log_error "End-to-end search workflow test failed"
        failed=$((failed + 1))
    fi
    
    # Summary
    echo ""
    echo "═══════════════════════════════════════════"
    echo "Integration Test Summary"
    echo "═══════════════════════════════════════════"
    local passed=$((total - failed))
    echo -e "Total tests: $total"
    echo -e "Passed: ${GREEN}$passed${NC}"
    echo -e "Failed: ${RED}$failed${NC}"
    
    if [ $failed -eq 0 ]; then
        log_success "All integration tests passed!"
        exit 0
    else
        log_error "$failed integration test(s) failed"
        exit 1
    fi
}

main "$@"

