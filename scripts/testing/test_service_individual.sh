#!/bin/bash
################################################################################
# Test Individual Service
# Tests a single service in isolation with its dependencies
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

check_port() {
    local port=$1
    timeout 2 bash -c "cat < /dev/null > /dev/tcp/localhost/$port" 2>/dev/null
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
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    log_success "$name is ready"
    return 0
}

check_http_health() {
    local url=$1
    local timeout=${2:-10}
    
    if timeout "$timeout" curl -sf "$url" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

wait_for_health() {
    local url=$1
    local timeout=${2:-60}
    local name=${3:-"service"}
    
    log_info "Waiting for $name health endpoint..."
    
    local elapsed=0
    while ! check_http_health "$url" 5; do
        if [ $elapsed -ge $timeout ]; then
            log_error "Timeout waiting for $name health"
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    log_success "$name health check passed"
    return 0
}

################################################################################
# Service Test Functions
################################################################################

test_redis() {
    log_step "Testing Redis"
    
    # Check if Redis is running in Docker
    if command -v docker &> /dev/null; then
        if docker ps --filter "name=redis" --filter "status=running" --format '{{.Names}}' | grep -q redis; then
            log_info "Redis container is running"
            
            # Try to ping via Docker exec
            if docker exec redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
                log_success "Redis is responding (via Docker)"
                return 0
            fi
        fi
    fi
    
    # Fallback to direct port check
    if check_port 6379; then
        if command -v redis-cli &> /dev/null; then
            if redis-cli -h localhost -p 6379 ping 2>/dev/null | grep -q "PONG"; then
                log_success "Redis is responding"
                return 0
            else
                log_error "Redis ping failed"
                return 1
            fi
        else
            log_success "Redis port is accessible"
            return 0
        fi
    else
        log_error "Redis not running on port 6379"
        return 1
    fi
}

test_postgres() {
    log_step "Testing PostgreSQL"
    
    # Check if PostgreSQL is running in Docker
    if command -v docker &> /dev/null; then
        if docker ps --filter "name=postgres" --filter "status=running" --format '{{.Names}}' | grep -q postgres; then
            log_info "PostgreSQL container is running"
            
            # Try to check via Docker exec
            if docker exec postgres pg_isready -U postgres >/dev/null 2>&1; then
                log_success "PostgreSQL is ready (via Docker)"
                return 0
            fi
        fi
    fi
    
    # Fallback to direct port check
    if check_port 5432; then
        if command -v pg_isready &> /dev/null; then
            if pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; then
                log_success "PostgreSQL is accepting connections"
                return 0
            else
                log_error "PostgreSQL not ready"
                return 1
            fi
        else
            log_success "PostgreSQL port is accessible"
            return 0
        fi
    else
        log_error "PostgreSQL not running on port 5432"
        return 1
    fi
}

test_neo4j() {
    log_step "Testing Neo4j"
    
    # Check if Neo4j is running in Docker
    if command -v docker &> /dev/null; then
        if docker ps --filter "name=neo4j" --filter "status=running" --format '{{.Names}}' | grep -q neo4j; then
            log_info "Neo4j container is running"
            
            # Try to check via cypher-shell (more reliable than curl in Neo4j container)
            if docker exec neo4j sh -c 'echo "RETURN 1" | cypher-shell -u neo4j -p amodels123 >/dev/null 2>&1' 2>/dev/null; then
                log_success "Neo4j is responding (via Docker cypher-shell)"
                
                # Check if we can access HTTP from host
                if check_http_health "http://localhost:7474" 5; then
                    log_success "Neo4j HTTP interface accessible from host"
                else
                    log_warn "Neo4j HTTP not accessible from host (may be network config)"
                fi
                return 0
            # Fallback: try curl if available
            elif docker exec neo4j curl -sf http://localhost:7474 >/dev/null 2>&1; then
                log_success "Neo4j HTTP interface accessible (via Docker)"
                return 0
            fi
        fi
    fi
    
    # Fallback to direct port check
    # Check HTTP port
    if ! check_port 7474; then
        log_error "Neo4j HTTP not running on port 7474"
        return 1
    fi
    
    # Check Bolt port
    if ! check_port 7687; then
        log_error "Neo4j Bolt not running on port 7687"
        return 1
    fi
    
    # Check HTTP endpoint
    if check_http_health "http://localhost:7474" 5; then
        log_success "Neo4j HTTP interface accessible"
    else
        log_error "Neo4j HTTP interface not accessible"
        return 1
    fi
    
    return 0
}

test_elasticsearch() {
    log_step "Testing Elasticsearch"
    
    # Check if Elasticsearch is running in Docker
    if command -v docker &> /dev/null; then
        if docker ps --filter "name=elasticsearch" --filter "status=running" --format '{{.Names}}' | grep -q elasticsearch; then
            log_info "Elasticsearch container is running"
            
            # Try to check health via Docker exec
            local es_health=$(docker exec elasticsearch curl -sf http://localhost:9200/_cluster/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "")
            if [ -n "$es_health" ]; then
                log_success "Elasticsearch cluster health: $es_health (via Docker)"
                
                # Check if accessible from host
                if check_http_health "http://localhost:9200/_cluster/health" 10; then
                    log_success "Elasticsearch accessible from host"
                else
                    log_warn "Elasticsearch not accessible from host (may be network config)"
                fi
                return 0
            fi
        fi
    fi
    
    # Fallback to direct port check
    if ! check_port 9200; then
        log_error "Elasticsearch not running on port 9200"
        return 1
    fi
    
    if check_http_health "http://localhost:9200/_cluster/health" 10; then
        local health=$(curl -sf http://localhost:9200/_cluster/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        log_success "Elasticsearch cluster health: $health"
        return 0
    else
        log_error "Elasticsearch health check failed"
        return 1
    fi
}

test_gitea() {
    log_step "Testing Gitea"
    
    # Check if Gitea is running in Docker
    if command -v docker &> /dev/null; then
        if docker ps --filter "name=gitea" --filter "status=running" --format '{{.Names}}' | grep -q gitea; then
            log_info "Gitea container is running"
            
            # Try to check health via Docker exec
            if docker exec gitea curl -sf http://localhost:3000/api/healthz >/dev/null 2>&1; then
                log_success "Gitea is healthy (via Docker)"
                
                # Check if accessible from host
                if check_http_health "http://localhost:3003/api/healthz" 10; then
                    log_success "Gitea accessible from host"
                else
                    log_warn "Gitea not accessible from host (may be network config)"
                fi
                return 0
            fi
        fi
    fi
    
    # Fallback to direct port check
    if ! check_port 3003; then
        log_error "Gitea not running on port 3003"
        return 1
    fi
    
    if check_http_health "http://localhost:3003/api/healthz" 10; then
        log_success "Gitea is healthy"
        return 0
    else
        log_error "Gitea health check failed"
        return 1
    fi
}

test_localai() {
    log_step "Testing LocalAI"
    
    if ! check_port 8081; then
        log_error "LocalAI not running on port 8081"
        return 1
    fi
    
    if check_http_health "http://localhost:8081/healthz" 10; then
        log_success "LocalAI health check passed"
        
        # Check models endpoint
        if check_http_health "http://localhost:8081/v1/models" 10; then
            log_success "LocalAI models endpoint accessible"
        else
            log_warn "LocalAI models endpoint not accessible"
        fi
        return 0
    else
        log_error "LocalAI health check failed"
        return 1
    fi
}

test_catalog() {
    log_step "Testing Catalog"
    
    if ! check_port 8084; then
        log_error "Catalog not running on port 8084"
        return 1
    fi
    
    if check_http_health "http://localhost:8084/health" 10; then
        log_success "Catalog is healthy"
        return 0
    else
        log_error "Catalog health check failed"
        return 1
    fi
}

test_extract() {
    log_step "Testing Extract"
    
    if ! check_port 8083; then
        log_error "Extract not running on port 8083"
        return 1
    fi
    
    if check_http_health "http://localhost:8083/health" 10; then
        log_success "Extract is healthy"
        return 0
    else
        log_error "Extract health check failed"
        return 1
    fi
}

test_graph() {
    log_step "Testing Graph"
    
    if ! check_port 8080; then
        log_error "Graph not running on port 8080"
        return 1
    fi
    
    if check_http_health "http://localhost:8080/health" 10; then
        log_success "Graph is healthy"
        return 0
    else
        log_error "Graph health check failed"
        return 1
    fi
}

test_search() {
    log_step "Testing Search"
    
    if ! check_port 8090; then
        log_error "Search not running on port 8090"
        return 1
    fi
    
    if check_http_health "http://localhost:8090/health" 10; then
        log_success "Search is healthy"
        return 0
    else
        log_error "Search health check failed"
        return 1
    fi
}

test_deepagents() {
    log_step "Testing DeepAgents"
    
    if ! check_port 9004; then
        log_error "DeepAgents not running on port 9004"
        return 1
    fi
    
    if check_http_health "http://localhost:9004/healthz" 10; then
        log_success "DeepAgents is healthy"
        return 0
    else
        log_error "DeepAgents health check failed"
        return 1
    fi
}

test_runtime() {
    log_step "Testing Runtime"
    
    if ! check_port 8098; then
        log_error "Runtime not running on port 8098"
        return 1
    fi
    
    if check_http_health "http://localhost:8098/healthz" 10; then
        log_success "Runtime is healthy"
        return 0
    else
        log_error "Runtime health check failed"
        return 1
    fi
}

test_orchestration() {
    log_step "Testing Orchestration"
    
    if ! check_port 8085; then
        log_error "Orchestration not running on port 8085"
        return 1
    fi
    
    if check_http_health "http://localhost:8085/healthz" 10; then
        log_success "Orchestration is healthy"
        return 0
    else
        log_error "Orchestration health check failed"
        return 1
    fi
}

test_training() {
    log_step "Testing Training"
    
    if ! check_port 8087; then
        log_error "Training not running on port 8087"
        return 1
    fi
    
    if check_http_health "http://localhost:8087/health" 10; then
        log_success "Training is healthy"
        return 0
    else
        log_error "Training health check failed"
        return 1
    fi
}

test_regulatory_audit() {
    log_step "Testing Regulatory Audit"
    
    if ! check_port 8099; then
        log_error "Regulatory Audit not running on port 8099"
        return 1
    fi
    
    if check_http_health "http://localhost:8099/healthz" 10; then
        log_success "Regulatory Audit is healthy"
        return 0
    else
        log_error "Regulatory Audit health check failed"
        return 1
    fi
}

test_telemetry_exporter() {
    log_step "Testing Telemetry Exporter"
    
    # Try different ports (8080, 8083, 8085)
    local ports=(8080 8083 8085)
    local found=false
    
    for port in "${ports[@]}"; do
        if check_port "$port"; then
            if check_http_health "http://localhost:$port/health" 10; then
                log_success "Telemetry Exporter is healthy on port $port"
                found=true
                break
            fi
        fi
    done
    
    if [ "$found" = false ]; then
        log_error "Telemetry Exporter not found on any expected port (8080, 8083, 8085)"
        return 1
    fi
    
    return 0
}

test_gateway() {
    log_step "Testing Gateway"
    
    if ! check_port 8000; then
        log_error "Gateway not running on port 8000"
        return 1
    fi
    
    if check_http_health "http://localhost:8000/healthz" 10; then
        log_success "Gateway is healthy"
        return 0
    else
        log_error "Gateway health check failed"
        return 1
    fi
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
        echo "  redis, postgres, neo4j, elasticsearch, gitea"
        echo "  localai, catalog, transformers"
        echo "  extract, graph, search, deepagents, runtime, orchestration, training, regulatory_audit, telemetry_exporter, gateway"
        exit 1
    fi
    
    log_info "Testing service: $service"
    
    case "$service" in
        redis)
            test_redis
            ;;
        postgres|postgresql)
            test_postgres
            ;;
        neo4j)
            test_neo4j
            ;;
        elasticsearch)
            test_elasticsearch
            ;;
        gitea)
            test_gitea
            ;;
        localai)
            test_localai
            ;;
        catalog)
            test_catalog
            ;;
        extract)
            test_extract
            ;;
        graph)
            test_graph
            ;;
        search)
            test_search
            ;;
        deepagents)
            test_deepagents
            ;;
        runtime)
            test_runtime
            ;;
        orchestration)
            test_orchestration
            ;;
        training)
            test_training
            ;;
        regulatory_audit|regulatory)
            test_regulatory_audit
            ;;
        telemetry_exporter|telemetry)
            test_telemetry_exporter
            ;;
        gateway)
            test_gateway
            ;;
        *)
            log_error "Unknown service: $service"
            exit 1
            ;;
    esac
}

main "$@"

