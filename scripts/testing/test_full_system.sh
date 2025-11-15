#!/bin/bash
################################################################################
# Full System Test
# Starts all services together and runs comprehensive system tests
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
START_SCRIPT="${PROJECT_ROOT}/scripts/system/start-system.sh"
HEALTH_SCRIPT="${PROJECT_ROOT}/scripts/system/health-check.sh"
TEST_INDIVIDUAL="${SCRIPT_DIR}/test_all_services_individual.sh"
TEST_FUNCTIONAL="${SCRIPT_DIR}/test_service_functionality.sh"
TEST_INTEGRATION="${SCRIPT_DIR}/test_service_integration.sh"
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
# System Test Functions
################################################################################

check_port_conflicts() {
    log_step "Checking for Port Conflicts"
    
    local ports=(
        "6379:Redis"
        "5432:PostgreSQL"
        "7474:Neo4j HTTP"
        "7687:Neo4j Bolt"
        "9200:Elasticsearch"
        "3003:Gitea"
        "8081:LocalAI"
        "8084:Catalog"
        "8083:Extract"
        "8080:Graph"
        "8090:Search"
        "9004:DeepAgents"
        "8098:Runtime"
        "8085:Orchestration"
        "8087:Training"
        "8099:Regulatory Audit"
        "8000:Gateway"
        "50051:PostgreSQL Lang"
    )
    
    local conflicts=0
    
    for port_info in "${ports[@]}"; do
        local port="${port_info%%:*}"
        local service="${port_info#*:}"
        
        # Count processes using this port
        local count=0
        if command -v lsof &> /dev/null; then
            count=$(lsof -Pi :"$port" -sTCP:LISTEN 2>/dev/null | wc -l)
        elif command -v netstat &> /dev/null; then
            count=$(netstat -tlnp 2>/dev/null | grep -c ":$port " || true)
        elif command -v ss &> /dev/null; then
            count=$(ss -tlnp 2>/dev/null | grep -c ":$port " || true)
        fi
        
        if [ "$count" -gt 1 ]; then
            log_warn "Port $port ($service) has $count listeners (potential conflict)"
            conflicts=$((conflicts + 1))
        elif [ "$count" -eq 1 ]; then
            log_info "Port $port ($service) is in use (normal)"
        fi
    done
    
    if [ $conflicts -eq 0 ]; then
        log_success "No port conflicts detected"
        return 0
    else
        log_warn "$conflicts potential port conflict(s) detected"
        return 1
    fi
}

check_system_resources() {
    log_step "Checking System Resources"
    
    # Check memory
    if command -v free &> /dev/null; then
        local mem_available=$(free -m | awk '/^Mem:/ {print $7}')
        local mem_total=$(free -m | awk '/^Mem:/ {print $2}')
        local mem_percent=$((mem_available * 100 / mem_total))
        
        log_info "Memory: ${mem_available}MB available / ${mem_total}MB total (${mem_percent}% free)"
        
        if [ $mem_percent -lt 10 ]; then
            log_warn "Low memory available (< 10%)"
        fi
    fi
    
    # Check disk space
    local disk_usage=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $5}' | sed 's/%//')
    log_info "Disk usage: ${disk_usage}%"
    
    if [ "$disk_usage" -gt 90 ]; then
        log_warn "Disk usage is high (> 90%)"
    fi
    
    # Check Docker (if available)
    if command -v docker &> /dev/null; then
        local docker_containers=$(docker ps --format '{{.Names}}' 2>/dev/null | wc -l)
        log_info "Docker containers running: $docker_containers"
    fi
    
    return 0
}

start_all_services() {
    log_step "Starting All Services"
    
    if [ ! -f "$START_SCRIPT" ]; then
        log_error "Start script not found: $START_SCRIPT"
        return 1
    fi
    
    log_info "Using start script: $START_SCRIPT"
    log_info "This may take several minutes..."
    
    # Start services with full profile
    if PROFILE=full "$START_SCRIPT" start > "${LOG_DIR}/startup.log" 2>&1; then
        log_success "Services started successfully"
        
        # Wait for services to be ready
        log_info "Waiting for services to be healthy (60 seconds)..."
        sleep 60
        
        return 0
    else
        log_error "Failed to start services (see ${LOG_DIR}/startup.log)"
        return 1
    fi
}

run_health_checks() {
    log_step "Running Comprehensive Health Checks"
    
    if [ ! -f "$HEALTH_SCRIPT" ]; then
        log_error "Health check script not found: $HEALTH_SCRIPT"
        return 1
    fi
    
    if "$HEALTH_SCRIPT" > "${LOG_DIR}/health_check.log" 2>&1; then
        log_success "Health checks passed"
        return 0
    else
        log_warn "Some health checks failed (see ${LOG_DIR}/health_check.log)"
        return 1
    fi
}

run_individual_tests() {
    log_step "Running Individual Service Tests"
    
    if [ ! -f "$TEST_INDIVIDUAL" ]; then
        log_warn "Individual test script not found, skipping"
        return 0
    fi
    
    if "$TEST_INDIVIDUAL" > "${LOG_DIR}/individual_tests.log" 2>&1; then
        log_success "Individual service tests passed"
        return 0
    else
        log_warn "Some individual service tests failed (see ${LOG_DIR}/individual_tests.log)"
        return 1
    fi
}

run_functional_tests() {
    log_step "Running Functional Tests"
    
    if [ ! -f "$TEST_FUNCTIONAL" ]; then
        log_warn "Functional test script not found, skipping"
        return 0
    fi
    
    local services=("localai" "catalog" "extract" "graph" "search" "deepagents" "runtime" "orchestration" "training" "regulatory_audit" "gateway" "telemetry_exporter")
    local passed=0
    local failed=0
    
    for service in "${services[@]}"; do
        if "$TEST_FUNCTIONAL" "$service" >> "${LOG_DIR}/functional_tests.log" 2>&1; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    log_info "Functional tests: $passed passed, $failed failed"
    
    if [ $failed -eq 0 ]; then
        log_success "All functional tests passed"
        return 0
    else
        log_warn "Some functional tests failed"
        return 1
    fi
}

run_integration_tests() {
    log_step "Running Integration Tests"
    
    if [ ! -f "$TEST_INTEGRATION" ]; then
        log_warn "Integration test script not found, skipping"
        return 0
    fi
    
    if "$TEST_INTEGRATION" > "${LOG_DIR}/integration_tests.log" 2>&1; then
        log_success "Integration tests passed"
        return 0
    else
        log_warn "Some integration tests failed (see ${LOG_DIR}/integration_tests.log)"
        return 1
    fi
}

################################################################################
# Main
################################################################################

main() {
    local start_services="${1:-yes}"
    
    log_header "Full System Test"
    log_info "Test logs will be written to: $LOG_DIR"
    echo ""
    
    # Pre-flight checks
    check_port_conflicts
    check_system_resources
    
    # Start services if requested
    if [ "$start_services" = "yes" ]; then
        if ! start_all_services; then
            log_error "Failed to start services, aborting tests"
            exit 1
        fi
    else
        log_info "Skipping service startup (assuming services are already running)"
    fi
    
    # Run test suite
    local test_results=0
    
    if ! run_health_checks; then
        test_results=$((test_results + 1))
    fi
    
    if ! run_individual_tests; then
        test_results=$((test_results + 1))
    fi
    
    if ! run_functional_tests; then
        test_results=$((test_results + 1))
    fi
    
    if ! run_integration_tests; then
        test_results=$((test_results + 1))
    fi
    
    # Summary
    log_header "Test Summary"
    log_info "All test logs available in: $LOG_DIR"
    
    if [ $test_results -eq 0 ]; then
        log_success "All system tests passed!"
        exit 0
    else
        log_error "$test_results test suite(s) had failures"
        exit 1
    fi
}

main "$@"

