#!/bin/bash
################################################################################
# Test All Services Individually
# Tests each service one by one to verify they can start and are healthy
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
TEST_SCRIPT="${SCRIPT_DIR}/test_service_individual.sh"
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

log_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"
}

################################################################################
# Test Results Tracking
################################################################################

declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_service_test() {
    local service=$1
    local category=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    log_info "Testing $service ($category)..."
    
    if "$TEST_SCRIPT" "$service" > "${LOG_DIR}/${service}.log" 2>&1; then
        TEST_RESULTS["$service"]="PASS"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        log_success "$service test passed"
        return 0
    else
        TEST_RESULTS["$service"]="FAIL"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        log_error "$service test failed (see ${LOG_DIR}/${service}.log)"
        return 1
    fi
}

################################################################################
# Main Test Execution
################################################################################

main() {
    log_header "Individual Service Testing"
    
    log_info "Test logs will be written to: $LOG_DIR"
    log_info "Using test script: $TEST_SCRIPT"
    echo ""
    
    # Infrastructure Services
    log_header "Infrastructure Services"
    run_service_test "redis" "infrastructure"
    run_service_test "postgres" "infrastructure"
    run_service_test "neo4j" "infrastructure"
    run_service_test "elasticsearch" "infrastructure"
    run_service_test "gitea" "infrastructure"
    
    # Core Services
    log_header "Core Services"
    run_service_test "localai" "core"
    run_service_test "catalog" "core"
    # transformers - optional, GPU required
    
    # Application Services
    log_header "Application Services"
    run_service_test "extract" "application"
    run_service_test "graph" "application"
    run_service_test "search" "application"
    run_service_test "deepagents" "application"
    run_service_test "runtime" "application"
    run_service_test "orchestration" "application"
    run_service_test "training" "application"
    run_service_test "regulatory_audit" "application"
    run_service_test "telemetry_exporter" "application"
    run_service_test "gateway" "application"
    
    # Summary
    log_header "Test Summary"
    echo "Total tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo ""
    
    # Detailed results
    echo "Service Results:"
    for service in "${!TEST_RESULTS[@]}"; do
        local result="${TEST_RESULTS[$service]}"
        if [ "$result" = "PASS" ]; then
            echo -e "  ${GREEN}✓${NC} $service"
        else
            echo -e "  ${RED}✗${NC} $service"
        fi
    done
    
    echo ""
    log_info "Detailed logs available in: $LOG_DIR"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "All service tests passed!"
        exit 0
    else
        log_error "$FAILED_TESTS service test(s) failed"
        exit 1
    fi
}

main "$@"

