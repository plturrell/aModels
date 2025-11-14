#!/bin/bash
################################################################################
# Start All Services - Comprehensive Startup Script
# 
# Builds LocalAI, starts all services in correct dependency order,
# performs health checks, and reports any errors or warnings.
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml"
BUILD_SCRIPT="${SCRIPT_DIR}/build-localai.sh"

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color
readonly BOLD='\033[1m'

# Service groups (in startup order)
readonly INFRASTRUCTURE="redis postgres neo4j elasticsearch"
readonly CORE="model-server models-storage transformers localai config-sync localai-compat"
readonly APP="catalog extract graph search-inference deepagents"

# Logging
log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_step() {
    echo -e "\n${BOLD}${BLUE}==>${NC} ${BOLD}$*${NC}"
}

log_header() {
    echo -e "\n${BOLD}${MAGENTA}════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${MAGENTA}  $*${NC}"
    echo -e "${BOLD}${MAGENTA}════════════════════════════════════════════════════════${NC}\n"
}

################################################################################
# Validation
################################################################################

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        return 1
    fi
    
    log_info "Docker is available and running"
    return 0
}

check_docker_compose() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        return 1
    fi
    
    # Validate docker-compose syntax
    if ! docker-compose -f "$COMPOSE_FILE" config &> /dev/null; then
        log_error "Docker Compose file has syntax errors"
        docker-compose -f "$COMPOSE_FILE" config 2>&1 | head -20
        return 1
    fi
    
    log_info "Docker Compose file is valid"
    return 0
}

check_build_script() {
    if [ ! -f "$BUILD_SCRIPT" ]; then
        log_error "Build script not found: $BUILD_SCRIPT"
        return 1
    fi
    
    if [ ! -x "$BUILD_SCRIPT" ]; then
        log_warn "Build script is not executable, making it executable..."
        chmod +x "$BUILD_SCRIPT"
    fi
    
    log_info "Build script is available"
    return 0
}

check_ports() {
    local ports=("6379" "5432" "7474" "7687" "9200" "8081" "8088" "9090" "8084" "8083" "8080" "8090")
    local conflicts=()
    
    for port in "${ports[@]}"; do
        if command -v lsof &> /dev/null; then
            if lsof -Pi :"$port" -sTCP:LISTEN -t &> /dev/null; then
                conflicts+=("$port")
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
                conflicts+=("$port")
            fi
        fi
    done
    
    if [ ${#conflicts[@]} -gt 0 ]; then
        log_warn "Port conflicts detected on: ${conflicts[*]}"
        log_warn "Some services may fail to start. Consider stopping conflicting services."
        return 1
    fi
    
    log_info "No port conflicts detected"
    return 0
}

check_volumes() {
    # Check if required directories exist (warnings only, not errors)
    local required_dirs=(
        "${PROJECT_ROOT}/services/localai/config"
    )
    
    local missing=()
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            missing+=("$dir")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_warn "Some directories are missing (may cause issues):"
        for dir in "${missing[@]}"; do
            log_warn "  - $dir"
        done
    fi
    
    return 0
}

################################################################################
# Service Management
################################################################################

stop_all_services() {
    log_step "Stopping any running services..."
    
    if docker-compose -f "$COMPOSE_FILE" ps -q | grep -q .; then
        log_info "Stopping existing services..."
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>&1 | grep -v "^WARNING:" || true
        sleep 2
        log_success "Services stopped"
    else
        log_info "No running services found"
    fi
}

build_localai() {
    log_step "Building LocalAI Docker image..."
    
    if [ ! -f "$BUILD_SCRIPT" ]; then
        log_error "Build script not found: $BUILD_SCRIPT"
        return 1
    fi
    
    if ! "$BUILD_SCRIPT"; then
        log_error "LocalAI build failed"
        return 1
    fi
    
    log_success "LocalAI build completed"
    return 0
}

start_services() {
    local services="$*"
    local service_list="${services// /, }"
    
    log_step "Starting services: $service_list"
    
    if ! docker-compose -f "$COMPOSE_FILE" up -d $services 2>&1 | grep -v "^WARNING:"; then
        log_error "Failed to start services: $services"
        return 1
    fi
    
    log_success "Services started: $service_list"
    return 0
}

wait_for_health() {
    local service=$1
    local max_wait=${2:-60}
    local health_url=${3:-}
    
    log_info "Waiting for $service to be healthy (max ${max_wait}s)..."
    
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        # Check container status
        local status=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null | xargs -r docker inspect --format='{{.State.Status}}' 2>/dev/null || echo "notfound")
        
        if [ "$status" = "running" ]; then
            # If health URL provided, check it
            if [ -n "$health_url" ]; then
                if curl -sf "$health_url" &> /dev/null; then
                    log_success "$service is healthy"
                    return 0
                fi
            else
                # Just check if container is running
                log_success "$service is running"
                return 0
            fi
        elif [ "$status" = "exited" ] || [ "$status" = "dead" ]; then
            log_error "$service container exited or died"
            docker-compose -f "$COMPOSE_FILE" logs --tail=20 "$service" 2>&1 | tail -10
            return 1
        fi
        
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    log_warn "$service did not become healthy within ${max_wait}s"
    docker-compose -f "$COMPOSE_FILE" logs --tail=20 "$service" 2>&1 | tail -10
    return 1
}

start_infrastructure() {
    log_step "Starting Infrastructure Services"
    
    if ! start_services $INFRASTRUCTURE; then
        return 1
    fi
    
    # Wait for critical infrastructure services
    log_info "Waiting for infrastructure services to be ready..."
    sleep 10
    
    # Check Redis
    wait_for_health "redis" 30 || log_warn "Redis may not be fully ready"
    
    # Check PostgreSQL
    wait_for_health "postgres" 60 || log_warn "PostgreSQL may not be fully ready"
    
    # Check Neo4j
    wait_for_health "neo4j" 60 || log_warn "Neo4j may not be fully ready"
    
    # Check Elasticsearch
    wait_for_health "elasticsearch" 60 "http://localhost:9200" || log_warn "Elasticsearch may not be fully ready"
    
    log_success "Infrastructure services started"
    return 0
}

start_core() {
    log_step "Starting Core Services"
    
    if ! start_services $CORE; then
        return 1
    fi
    
    log_info "Waiting for core services to be ready..."
    sleep 15
    
    # Check model-server
    wait_for_health "model-server" 30 "http://localhost:8088/health" || log_warn "model-server may not be fully ready"
    
    # Check transformers
    wait_for_health "transformers-service" 60 "http://localhost:9090/health" || log_warn "transformers may not be fully ready"
    
    # Check localai
    wait_for_health "localai" 60 "http://localhost:8081/healthz" || log_warn "localai may not be fully ready"
    
    log_success "Core services started"
    return 0
}

start_app() {
    log_step "Starting Application Services"
    
    if ! start_services $APP; then
        return 1
    fi
    
    log_info "Waiting for application services to be ready..."
    sleep 10
    
    log_success "Application services started"
    return 0
}

perform_health_checks() {
    log_step "Performing Health Checks"
    
    local failed=0
    local services=(
        "redis:6379"
        "postgres:5432"
        "neo4j:7474"
        "elasticsearch:9200"
        "model-server:8088"
        "transformers-service:9090"
        "localai:8081"
        "catalog:8084"
    )
    
    for service_port in "${services[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        if docker-compose -f "$COMPOSE_FILE" ps -q "$service" | grep -q .; then
            if command -v curl &> /dev/null; then
                if curl -sf "http://localhost:$port" &> /dev/null || curl -sf "http://localhost:$port/health" &> /dev/null || curl -sf "http://localhost:$port/healthz" &> /dev/null; then
                    log_success "$service (port $port) is responding"
                else
                    log_warn "$service (port $port) is not responding to health checks"
                    failed=$((failed + 1))
                fi
            else
                log_info "$service (port $port) - curl not available, skipping HTTP check"
            fi
        else
            log_warn "$service container not found"
            failed=$((failed + 1))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        log_success "All health checks passed"
        return 0
    else
        log_warn "$failed service(s) failed health checks"
        return 1
    fi
}

show_status() {
    log_step "Service Status"
    docker-compose -f "$COMPOSE_FILE" ps
}

################################################################################
# Main
################################################################################

main() {
    local errors=0
    local warnings=0
    
    log_header "aModels Service Startup"
    
    # Validation
    log_step "Pre-flight Checks"
    if ! check_docker; then
        errors=$((errors + 1))
    fi
    if ! check_docker_compose; then
        errors=$((errors + 1))
    fi
    if ! check_build_script; then
        errors=$((errors + 1))
    fi
    check_ports || warnings=$((warnings + 1))
    check_volumes || warnings=$((warnings + 1))
    
    if [ $errors -gt 0 ]; then
        log_error "Pre-flight checks failed. Please fix errors before continuing."
        exit 1
    fi
    
    if [ $warnings -gt 0 ]; then
        log_warn "Pre-flight checks completed with $warnings warning(s)"
    else
        log_success "Pre-flight checks passed"
    fi
    
    # Stop existing services
    stop_all_services
    
    # Build LocalAI
    if ! build_localai; then
        log_error "Failed to build LocalAI. Aborting startup."
        exit 1
    fi
    
    # Start services in order
    if ! start_infrastructure; then
        log_error "Failed to start infrastructure services"
        errors=$((errors + 1))
    fi
    
    if ! start_core; then
        log_error "Failed to start core services"
        errors=$((errors + 1))
    fi
    
    if ! start_app; then
        log_error "Failed to start application services"
        errors=$((errors + 1))
    fi
    
    # Health checks
    if ! perform_health_checks; then
        warnings=$((warnings + 1))
    fi
    
    # Final status
    echo ""
    show_status
    echo ""
    
    if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
        log_success "All services started successfully with no errors or warnings!"
        exit 0
    elif [ $errors -eq 0 ]; then
        log_warn "Services started with $warnings warning(s). Check logs for details."
        exit 0
    else
        log_error "Startup completed with $errors error(s) and $warnings warning(s)"
        exit 1
    fi
}

main "$@"

