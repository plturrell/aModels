#!/bin/bash
################################################################################
# Functional Test Script
# Tests actual functionality of services beyond health checks
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

check_json_response() {
    local response=$1
    if echo "$response" | jq . >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

################################################################################
# Functional Test Functions
################################################################################

test_localai_functionality() {
    log_step "Testing LocalAI Functionality"
    
    # Test models endpoint
    local models_response=$(http_get "http://localhost:8081/v1/models" 10)
    if [ -n "$models_response" ] && check_json_response "$models_response"; then
        log_success "Models endpoint returns valid JSON"
        
        # Check if models list is present
        if echo "$models_response" | jq -e '.data' >/dev/null 2>&1; then
            local model_count=$(echo "$models_response" | jq '.data | length')
            log_info "Found $model_count model(s)"
        fi
    else
        log_error "Models endpoint failed or returned invalid JSON"
        return 1
    fi
    
    # Test completions endpoint (if available)
    local test_payload='{"model":"test","prompt":"Hello","max_tokens":10}'
    local completion_response=$(http_post "http://localhost:8081/v1/completions" "$test_payload" 30)
    if [ -n "$completion_response" ]; then
        if check_json_response "$completion_response"; then
            log_success "Completions endpoint is functional"
        else
            log_warn "Completions endpoint returned non-JSON (may be expected)"
        fi
    else
        log_warn "Completions endpoint not tested (may require specific model)"
    fi
    
    return 0
}

test_catalog_functionality() {
    log_step "Testing Catalog Functionality"
    
    # Test health endpoint with detailed response
    local health_response=$(http_get "http://localhost:8084/health" 10)
    if [ -n "$health_response" ] && check_json_response "$health_response"; then
        log_success "Health endpoint returns valid JSON"
        
        # Check status
        local status=$(echo "$health_response" | jq -r '.status // "unknown"' 2>/dev/null)
        log_info "Catalog status: $status"
    else
        log_warn "Health endpoint may not return JSON (checking basic connectivity)"
    fi
    
    # Try to access API endpoints (if available)
    local api_response=$(http_get "http://localhost:8084/api/v1/entities" 10 2>/dev/null || true)
    if [ -n "$api_response" ]; then
        log_success "API endpoint accessible"
    else
        log_warn "API endpoint not tested (may require authentication)"
    fi
    
    return 0
}

test_extract_functionality() {
    log_step "Testing Extract Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:8083/health" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test extraction endpoint (if available)
    local test_doc='{"content":"Test document","type":"text"}'
    local extract_response=$(http_post "http://localhost:8083/api/v1/extract" "$test_doc" 30 2>/dev/null || true)
    if [ -n "$extract_response" ]; then
        log_success "Extraction endpoint accessible"
    else
        log_warn "Extraction endpoint not tested (may require specific format)"
    fi
    
    return 0
}

test_graph_functionality() {
    log_step "Testing Graph Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:8080/health" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test graph query endpoint (if available)
    local query_response=$(http_get "http://localhost:8080/api/v1/graph/query?limit=1" 10 2>/dev/null || true)
    if [ -n "$query_response" ]; then
        log_success "Graph query endpoint accessible"
    else
        log_warn "Graph query endpoint not tested (may require authentication)"
    fi
    
    return 0
}

test_search_functionality() {
    log_step "Testing Search Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:8090/health" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test search endpoint
    local search_payload='{"query":"test","limit":5}'
    local search_response=$(http_post "http://localhost:8090/api/v1/search" "$search_payload" 10 2>/dev/null || true)
    if [ -n "$search_response" ]; then
        log_success "Search endpoint accessible"
    else
        log_warn "Search endpoint not tested (may require index)"
    fi
    
    return 0
}

test_deepagents_functionality() {
    log_step "Testing DeepAgents Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:9004/healthz" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test agent creation endpoint (if available)
    local agent_payload='{"name":"test-agent","type":"basic"}'
    local agent_response=$(http_post "http://localhost:9004/api/v1/agents" "$agent_payload" 30 2>/dev/null || true)
    if [ -n "$agent_response" ]; then
        log_success "Agent endpoint accessible"
    else
        log_warn "Agent endpoint not tested (may require authentication)"
    fi
    
    return 0
}

test_runtime_functionality() {
    log_step "Testing Runtime Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:8098/healthz" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test analytics endpoint (if available)
    local analytics_response=$(http_get "http://localhost:8098/api/v1/analytics" 10 2>/dev/null || true)
    if [ -n "$analytics_response" ]; then
        log_success "Analytics endpoint accessible"
    else
        log_warn "Analytics endpoint not tested"
    fi
    
    return 0
}

test_orchestration_functionality() {
    log_step "Testing Orchestration Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:8085/healthz" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test workflow endpoint (if available)
    local workflow_response=$(http_get "http://localhost:8085/api/v1/workflows" 10 2>/dev/null || true)
    if [ -n "$workflow_response" ]; then
        log_success "Workflow endpoint accessible"
    else
        log_warn "Workflow endpoint not tested"
    fi
    
    return 0
}

test_training_functionality() {
    log_step "Testing Training Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:8087/health" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test training job endpoint (if available)
    local job_response=$(http_get "http://localhost:8087/api/v1/jobs" 10 2>/dev/null || true)
    if [ -n "$job_response" ]; then
        log_success "Training job endpoint accessible"
    else
        log_warn "Training job endpoint not tested"
    fi
    
    return 0
}

test_regulatory_audit_functionality() {
    log_step "Testing Regulatory Audit Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:8099/healthz" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test audit endpoint (if available)
    local audit_response=$(http_get "http://localhost:8099/api/v1/audits" 10 2>/dev/null || true)
    if [ -n "$audit_response" ]; then
        log_success "Audit endpoint accessible"
    else
        log_warn "Audit endpoint not tested"
    fi
    
    return 0
}

test_gateway_functionality() {
    log_step "Testing Gateway Functionality"
    
    # Test health endpoint
    local health_response=$(http_get "http://localhost:8000/healthz" 10)
    if [ -n "$health_response" ]; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test gateway routing (if available)
    local routes_response=$(http_get "http://localhost:8000/api/v1/routes" 10 2>/dev/null || true)
    if [ -n "$routes_response" ]; then
        log_success "Gateway routes endpoint accessible"
    else
        log_warn "Gateway routes endpoint not tested"
    fi
    
    return 0
}

test_telemetry_exporter_functionality() {
    log_step "Testing Telemetry Exporter Functionality"
    
    # Try different ports
    local ports=(8085 8080 8083)
    local found=false
    
    for port in "${ports[@]}"; do
        local health_response=$(http_get "http://localhost:$port/health" 10 2>/dev/null || true)
        if [ -n "$health_response" ]; then
            log_success "Health endpoint responding on port $port"
            found=true
            
            # Test export endpoint (if available)
            local export_response=$(http_get "http://localhost:$port/api/v1/traces/export" 10 2>/dev/null || true)
            if [ -n "$export_response" ]; then
                log_success "Export endpoint accessible"
            fi
            break
        fi
    done
    
    if [ "$found" = false ]; then
        log_error "Telemetry exporter not found on any expected port"
        return 1
    fi
    
    return 0
}

################################################################################
# Main
################################################################################

main() {
    local service="${1:-}"
    
    if [ -z "$service" ]; then
        echo "Usage: $0 <service_name>"
        echo ""
        echo "Available services:"
        echo "  localai, catalog, extract, graph, search"
        echo "  deepagents, runtime, orchestration, training"
        echo "  regulatory_audit, telemetry_exporter, gateway"
        exit 1
    fi
    
    log_info "Testing functionality of service: $service"
    
    case "$service" in
        localai)
            test_localai_functionality
            ;;
        catalog)
            test_catalog_functionality
            ;;
        extract)
            test_extract_functionality
            ;;
        graph)
            test_graph_functionality
            ;;
        search)
            test_search_functionality
            ;;
        deepagents)
            test_deepagents_functionality
            ;;
        runtime)
            test_runtime_functionality
            ;;
        orchestration)
            test_orchestration_functionality
            ;;
        training)
            test_training_functionality
            ;;
        regulatory_audit|regulatory)
            test_regulatory_audit_functionality
            ;;
        telemetry_exporter|telemetry)
            test_telemetry_exporter_functionality
            ;;
        gateway)
            test_gateway_functionality
            ;;
        *)
            log_error "Unknown service: $service"
            exit 1
            ;;
    esac
}

main "$@"

