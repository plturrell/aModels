#!/bin/bash
################################################################################
# aModels System Startup Orchestrator
# 
# Robust startup process for all services with:
# - Dependency management
# - Health checking
# - Proper startup ordering
# - Graceful error handling
# - Multiple deployment modes (native, docker, hybrid)
################################################################################

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color
readonly BOLD='\033[1m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config/services.yaml"
LOG_DIR="${PROJECT_ROOT}/logs/startup"
PID_DIR="${PROJECT_ROOT}/.pids"
PROFILE="${PROFILE:-development}"
MODE="${MODE:-native}"  # native, docker, hybrid

# Create necessary directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# Load helper functions
source "${SCRIPT_DIR}/lib/service-utils.sh" 2>/dev/null || true

################################################################################
# Logging Functions
################################################################################

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
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
# Utility Functions
################################################################################

check_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1
    elif command -v netstat &> /dev/null; then
        netstat -tlnp 2>/dev/null | grep -q ":$port "
    elif command -v ss &> /dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":$port "
    else
        # Fallback: try to connect
        timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/$port" 2>/dev/null
    fi
}

wait_for_port() {
    local port=$1
    local timeout=${2:-60}
    local name=${3:-"service on port $port"}
    
    log_info "Waiting for $name (port $port)..."
    
    local elapsed=0
    while ! check_port "$port"; do
        if [ $elapsed -ge $timeout ]; then
            log_error "Timeout waiting for $name"
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    
    log_success "$name is ready"
    return 0
}

wait_for_health() {
    local url=$1
    local timeout=${2:-60}
    local name=${3:-"service at $url"}
    
    log_info "Checking health: $name..."
    
    local elapsed=0
    while ! curl -sf "$url" >/dev/null 2>&1; do
        if [ $elapsed -ge $timeout ]; then
            log_warn "Health check timeout for $name (non-fatal)"
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    
    log_success "$name health check passed"
    return 0
}

check_dependency() {
    local dep=$1
    if ! command -v "$dep" &> /dev/null; then
        log_error "Required dependency not found: $dep"
        return 1
    fi
    return 0
}

################################################################################
# Infrastructure Services
################################################################################

start_redis() {
    log_step "Starting Redis"
    
    if check_port 6379; then
        log_warn "Redis already running on port 6379"
        return 0
    fi
    
    if [ "$MODE" = "docker" ]; then
        docker-compose -f "${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml" up -d redis
    else
        # Native Redis
        if command -v redis-server &> /dev/null; then
            nohup redis-server --port 6379 --appendonly yes > "$LOG_DIR/redis.log" 2>&1 &
            echo $! > "$PID_DIR/redis.pid"
        else
            log_error "Redis not found. Install redis-server or use docker mode."
            return 1
        fi
    fi
    
    wait_for_port 6379 30 "Redis"
}

start_postgres() {
    log_step "Starting PostgreSQL"
    
    if check_port 5432; then
        log_warn "PostgreSQL already running on port 5432"
        return 0
    fi
    
    if [ "$MODE" = "docker" ]; then
        docker-compose -f "${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml" up -d postgres
        wait_for_port 5432 60 "PostgreSQL"
    else
        log_warn "Native PostgreSQL requires manual start. Using existing instance..."
    fi
    
    # Wait for PostgreSQL to be ready
    local retries=0
    while ! pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; do
        if [ $retries -ge 30 ]; then
            log_error "PostgreSQL not ready"
            return 1
        fi
        sleep 2
        retries=$((retries + 1))
    done
    
    log_success "PostgreSQL is ready"
}

start_neo4j() {
    log_step "Starting Neo4j"
    
    if check_port 7687; then
        log_warn "Neo4j already running on port 7687"
        return 0
    fi
    
    if [ "$MODE" = "docker" ]; then
        docker-compose -f "${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml" up -d neo4j
    else
        log_error "Neo4j requires Docker. Set MODE=docker or start manually."
        return 1
    fi
    
    wait_for_port 7474 90 "Neo4j HTTP"
    wait_for_port 7687 90 "Neo4j Bolt"
    
    # Wait for Neo4j to be fully ready
    sleep 10
    log_success "Neo4j is ready"
}

start_elasticsearch() {
    log_step "Starting Elasticsearch"
    
    if check_port 9200; then
        log_warn "Elasticsearch already running on port 9200"
        return 0
    fi
    
    if [ "$MODE" = "docker" ]; then
        docker-compose -f "${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml" up -d elasticsearch
        wait_for_port 9200 120 "Elasticsearch"
        wait_for_health "http://localhost:9200/_cluster/health" 120 "Elasticsearch"
    else
        log_warn "Elasticsearch requires Docker or manual setup"
        return 1
    fi
}

################################################################################
# Core Services
################################################################################

start_localai() {
    log_step "Starting LocalAI"
    
    if check_port 8081; then
        log_warn "LocalAI already running on port 8081"
        return 0
    fi
    
    cd "$PROJECT_ROOT/services/localai"
    
    if [ "$MODE" = "docker" ]; then
        docker-compose -f "${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml" up -d localai
    else
        # Native LocalAI (if available)
        if [ -f "cmd/local-ai/main.go" ]; then
            log_info "Building LocalAI..."
            go build -o ../../bin/localai ./cmd/local-ai
            nohup ../../bin/localai > "$LOG_DIR/localai.log" 2>&1 &
            echo $! > "$PID_DIR/localai.pid"
        else
            log_error "LocalAI source not found"
            return 1
        fi
    fi
    
    wait_for_port 8081 180 "LocalAI"
    wait_for_health "http://localhost:8081/healthz" 60 "LocalAI"
}

start_catalog() {
    log_step "Starting Catalog Service"
    
    if check_port 8084; then
        log_warn "Catalog already running on port 8084"
        return 0
    fi
    
    cd "$PROJECT_ROOT/services/catalog"
    
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USERNAME="neo4j"
    export NEO4J_PASSWORD="amodels123"
    export REDIS_URL="redis://localhost:6379/0"
    export POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/amodels?sslmode=disable"
    export PORT="8084"
    
    if [ "$MODE" = "docker" ]; then
        docker-compose -f "${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml" up -d catalog
    else
        log_info "Building Catalog service..."
        go build -o ../../bin/catalog .
        nohup ../../bin/catalog > "$LOG_DIR/catalog.log" 2>&1 &
        echo $! > "$PID_DIR/catalog.pid"
    fi
    
    wait_for_port 8084 90 "Catalog"
    wait_for_health "http://localhost:8084/health" 60 "Catalog"
}

################################################################################
# Application Services
################################################################################

start_extract() {
    log_step "Starting Extract Service"
    
    if check_port 8083; then
        log_warn "Extract already running on port 8083"
        return 0
    fi
    
    cd "$PROJECT_ROOT/services/extract"
    
    export PORT="8082"
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USERNAME="neo4j"
    export NEO4J_PASSWORD="amodels123"
    export POSTGRES_CATALOG_DSN="postgresql://postgres:postgres@localhost:5432/amodels?sslmode=disable"
    
    if [ "$MODE" = "docker" ]; then
        docker-compose -f "${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml" up -d extract
    else
        log_info "Building Extract service..."
        go build -o ../../bin/extract ./cmd/extract
        nohup ../../bin/extract > "$LOG_DIR/extract.log" 2>&1 &
        echo $! > "$PID_DIR/extract.pid"
    fi
    
    wait_for_port 8083 90 "Extract"
}

start_runtime() {
    log_step "Starting Runtime Analytics Server"
    
    if check_port 8098; then
        log_warn "Runtime already running on port 8098"
        return 0
    fi
    
    cd "$PROJECT_ROOT/services/runtime"
    
    export RUNTIME_ADDR=":8098"
    export CATALOG_URL="http://localhost:8084"
    
    log_info "Building Runtime service..."
    go build -o ../../bin/runtime ./cmd/server
    nohup ../../bin/runtime > "$LOG_DIR/runtime.log" 2>&1 &
    echo $! > "$PID_DIR/runtime.pid"
    
    wait_for_port 8098 60 "Runtime"
    wait_for_health "http://localhost:8098/healthz" 30 "Runtime"
}

start_orchestration() {
    log_step "Starting Orchestration Server"
    
    if check_port 8085; then
        log_warn "Orchestration already running on port 8085"
        return 0
    fi
    
    cd "$PROJECT_ROOT/services/orchestration"
    
    export ORCHESTRATION_PORT="8085"
    
    log_info "Building Orchestration service..."
    go build -o ../../bin/orchestration ./cmd/server
    nohup ../../bin/orchestration > "$LOG_DIR/orchestration.log" 2>&1 &
    echo $! > "$PID_DIR/orchestration.pid"
    
    wait_for_port 8085 60 "Orchestration"
    wait_for_health "http://localhost:8085/healthz" 30 "Orchestration"
}

start_regulatory_audit() {
    log_step "Starting Regulatory Audit Server"
    
    if check_port 8099; then
        log_warn "Regulatory Audit already running on port 8099"
        return 0
    fi
    
    cd "$PROJECT_ROOT/services/regulatory"
    
    export AUDIT_SERVER_ADDR=":8099"
    export NEO4J_URL="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="amodels123"
    export LOCALAI_URL="http://localhost:8081"
    export GNN_SERVICE_URL="http://localhost:8081"
    export GOOSE_SERVER_URL="http://localhost:8082"
    export DEEPAGENTS_URL="http://localhost:9004"
    
    log_info "Building Regulatory Audit service..."
    go build -o ../../bin/audit-server ./cmd/audit-server
    nohup ../../bin/audit-server > "$LOG_DIR/audit-server.log" 2>&1 &
    echo $! > "$PID_DIR/audit-server.pid"
    
    wait_for_port 8099 90 "Regulatory Audit"
    wait_for_health "http://localhost:8099/healthz" 30 "Regulatory Audit"
}

################################################################################
# Profile Management
################################################################################

get_services_for_profile() {
    case "$PROFILE" in
        minimal)
            echo "redis postgres neo4j localai catalog"
            ;;
        development)
            echo "redis postgres neo4j elasticsearch localai catalog extract runtime orchestration"
            ;;
        full)
            echo "redis postgres neo4j elasticsearch localai catalog extract runtime orchestration regulatory_audit"
            ;;
        docker)
            echo "docker-compose"
            ;;
        *)
            log_error "Unknown profile: $PROFILE"
            echo "redis postgres neo4j localai catalog"
            ;;
    esac
}

################################################################################
# Main Startup Logic
################################################################################

start_all_services() {
    local services=$(get_services_for_profile)
    
    log_header "Starting aModels System - Profile: $PROFILE, Mode: $MODE"
    
    if [ "$MODE" = "docker" ] || [ "$services" = "docker-compose" ]; then
        log_step "Starting all services via Docker Compose"
        cd "$PROJECT_ROOT"

        local compose_file="${PROJECT_ROOT}/infrastructure/docker/compose.yml"
        if [ ! -f "$compose_file" ]; then
            log_warn "${compose_file} not found, falling back to ${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml"
            compose_file="${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml"
        fi

        docker-compose -f "$compose_file" up -d
        
        log_info "Waiting for services to be healthy..."
        sleep 30
        
        log_success "Docker Compose services started"
        show_service_status
        return 0
    fi
    
    # Start services in dependency order
    for service in $services; do
        case "$service" in
            redis) start_redis || log_warn "Redis start failed (non-fatal)" ;;
            postgres) start_postgres || log_error "PostgreSQL required!" ;;
            neo4j) start_neo4j || log_error "Neo4j required!" ;;
            elasticsearch) start_elasticsearch || log_warn "Elasticsearch start failed (non-fatal)" ;;
            localai) start_localai || log_error "LocalAI required!" ;;
            catalog) start_catalog || log_error "Catalog required!" ;;
            extract) start_extract || log_warn "Extract start failed (non-fatal)" ;;
            runtime) start_runtime || log_warn "Runtime start failed (non-fatal)" ;;
            orchestration) start_orchestration || log_warn "Orchestration start failed (non-fatal)" ;;
            regulatory_audit) start_regulatory_audit || log_warn "Regulatory Audit start failed (non-fatal)" ;;
        esac
    done
    
    log_header "Startup Complete"
    show_service_status
}

show_service_status() {
    log_header "Service Status"
    
    local services=(
        "6379:Redis"
        "5432:PostgreSQL"
        "7687:Neo4j"
        "9200:Elasticsearch"
        "8081:LocalAI"
        "8084:Catalog"
        "8083:Extract"
        "8098:Runtime"
        "8085:Orchestration"
        "8099:Regulatory Audit"
    )
    
    for service in "${services[@]}"; do
        local port="${service%%:*}"
        local name="${service#*:}"
        
        if check_port "$port"; then
            echo -e "${GREEN}✓${NC} ${BOLD}$name${NC} - http://localhost:$port"
        else
            echo -e "${YELLOW}○${NC} ${BOLD}$name${NC} - not running"
        fi
    done
    
    echo ""
    log_info "Logs available in: $LOG_DIR"
    log_info "PIDs available in: $PID_DIR"
    echo ""
}

################################################################################
# Shutdown
################################################################################

shutdown_services() {
    log_header "Shutting Down Services"
    
    if [ "$MODE" = "docker" ]; then
        log_info "Stopping Docker Compose services..."
        cd "$PROJECT_ROOT"
        docker-compose -f infrastructure/docker/brev/docker-compose.yml down
        return 0
    fi
    
    # Stop PID-tracked services
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local name=$(basename "$pid_file" .pid)
            
            if ps -p "$pid" > /dev/null 2>&1; then
                log_info "Stopping $name (PID: $pid)"
                kill "$pid" 2>/dev/null || true
                sleep 2
                
                # Force kill if still running
                if ps -p "$pid" > /dev/null 2>&1; then
                    log_warn "Force killing $name"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
            
            rm "$pid_file"
        fi
    done
    
    log_success "All services stopped"
}

################################################################################
# Command Line Interface
################################################################################

show_help() {
    cat << EOF
aModels System Startup Orchestrator

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    start           Start all services (default)
    stop            Stop all services
    restart         Restart all services
    status          Show service status
    logs            Tail all service logs
    clean           Clean logs and PID files

Options:
    --profile=PROFILE   Startup profile: minimal, development, full, docker (default: development)
    --mode=MODE         Deployment mode: native, docker, hybrid (default: native)
    --help              Show this help message

Environment Variables:
    PROFILE             Same as --profile
    MODE                Same as --mode

Examples:
    $0 start --profile=minimal
    $0 start --mode=docker
    PROFILE=full MODE=docker $0 start
    $0 stop
    $0 status

EOF
}

main() {
    local command="${1:-start}"
    
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --profile=*)
                PROFILE="${arg#*=}"
                shift
                ;;
            --mode=*)
                MODE="${arg#*=}"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            start|stop|restart|status|logs|clean)
                command="$arg"
                shift
                ;;
        esac
    done
    
    case "$command" in
        start)
            start_all_services
            ;;
        stop)
            shutdown_services
            ;;
        restart)
            shutdown_services
            sleep 2
            start_all_services
            ;;
        status)
            show_service_status
            ;;
        logs)
            log_info "Tailing logs from $LOG_DIR"
            tail -f "$LOG_DIR"/*.log
            ;;
        clean)
            log_info "Cleaning logs and PID files"
            rm -rf "$LOG_DIR"/* "$PID_DIR"/*
            log_success "Cleaned"
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Trap signals for graceful shutdown
trap shutdown_services SIGINT SIGTERM EXIT

main "$@"
