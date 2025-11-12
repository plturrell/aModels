#!/bin/bash
################################################################################
# aModels System Health Check
# Comprehensive health checking for all services
################################################################################

set -euo pipefail

# Color codes
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Service definitions
declare -A SERVICES=(
    ["Redis"]="tcp:6379"
    ["PostgreSQL"]="tcp:5432"
    ["Neo4j HTTP"]="http:http://localhost:7474"
    ["Neo4j Bolt"]="tcp:7687"
    ["Elasticsearch"]="http:http://localhost:9200/_cluster/health"
    ["LocalAI"]="http:http://localhost:8081/healthz"
    ["Catalog"]="http:http://localhost:8084/health"
    ["Extract"]="http:http://localhost:8083/health"
    ["Graph"]="http:http://localhost:8080/health"
    ["Search"]="http:http://localhost:8090/health"
    ["DeepAgents"]="http:http://localhost:9004/healthz"
    ["Runtime"]="http:http://localhost:8098/healthz"
    ["Orchestration"]="http:http://localhost:8085/healthz"
    ["Training"]="http:http://localhost:8087/health"
    ["DMS"]="http:http://localhost:8096/docs"
    ["Regulatory Audit"]="http:http://localhost:8099/healthz"
)

################################################################################
# Health Check Functions
################################################################################

check_tcp_port() {
    local port=$1
    timeout 2 bash -c "cat < /dev/null > /dev/tcp/localhost/$port" 2>/dev/null
}

check_http_endpoint() {
    local url=$1
    timeout 5 curl -sf "$url" >/dev/null 2>&1
}

check_service() {
    local name=$1
    local check_type="${SERVICES[$name]%%:*}"
    local check_value="${SERVICES[$name]#*:}"
    
    case "$check_type" in
        tcp)
            if check_tcp_port "$check_value"; then
                echo -e "${GREEN}✓${NC} $name (port $check_value)"
                return 0
            else
                echo -e "${RED}✗${NC} $name (port $check_value)"
                return 1
            fi
            ;;
        http)
            if check_http_endpoint "$check_value"; then
                echo -e "${GREEN}✓${NC} $name"
                return 0
            else
                echo -e "${RED}✗${NC} $name"
                return 1
            fi
            ;;
        *)
            echo -e "${YELLOW}?${NC} $name (unknown check type)"
            return 1
            ;;
    esac
}

################################################################################
# Detailed Health Checks
################################################################################

check_postgres_detail() {
    echo ""
    echo "PostgreSQL Detailed Check:"
    
    if pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Server is accepting connections"
        
        # Check database exists
        if psql -h localhost -U postgres -lqt | cut -d \| -f 1 | grep -qw amodels; then
            echo -e "  ${GREEN}✓${NC} Database 'amodels' exists"
        else
            echo -e "  ${YELLOW}!${NC} Database 'amodels' not found"
        fi
    else
        echo -e "  ${RED}✗${NC} Server not responding"
    fi
}

check_neo4j_detail() {
    echo ""
    echo "Neo4j Detailed Check:"
    
    if curl -sf http://localhost:7474 >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} HTTP interface accessible"
    else
        echo -e "  ${RED}✗${NC} HTTP interface not accessible"
    fi
    
    # Try to run a simple query
    if command -v cypher-shell &> /dev/null; then
        if echo "RETURN 1" | cypher-shell -u neo4j -p amodels123 >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Cypher queries working"
        else
            echo -e "  ${YELLOW}!${NC} Could not execute Cypher query"
        fi
    fi
}

check_elasticsearch_detail() {
    echo ""
    echo "Elasticsearch Detailed Check:"
    
    if curl -sf http://localhost:9200 >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Server responding"
        
        # Check cluster health
        local health=$(curl -sf http://localhost:9200/_cluster/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        case "$health" in
            green)
                echo -e "  ${GREEN}✓${NC} Cluster health: $health"
                ;;
            yellow)
                echo -e "  ${YELLOW}!${NC} Cluster health: $health"
                ;;
            red)
                echo -e "  ${RED}✗${NC} Cluster health: $health"
                ;;
        esac
    else
        echo -e "  ${RED}✗${NC} Server not responding"
    fi
}

check_localai_detail() {
    echo ""
    echo "LocalAI Detailed Check:"
    
    if curl -sf http://localhost:8081/healthz >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Health endpoint responding"
        
        # Check models endpoint
        if curl -sf http://localhost:8081/v1/models >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Models endpoint accessible"
            
            # Count models
            local model_count=$(curl -sf http://localhost:8081/v1/models | grep -o '"id"' | wc -l)
            echo -e "  ${GREEN}✓${NC} Loaded models: $model_count"
        else
            echo -e "  ${YELLOW}!${NC} Models endpoint not accessible"
        fi
    else
        echo -e "  ${RED}✗${NC} Health endpoint not responding"
    fi
}

################################################################################
# System Health
################################################################################

check_system_resources() {
    echo ""
    echo "═══════════════════════════════════════════"
    echo "System Resources"
    echo "═══════════════════════════════════════════"
    
    # CPU
    if command -v mpstat &> /dev/null; then
        local cpu_idle=$(mpstat 1 1 | tail -1 | awk '{print $NF}')
        echo -e "CPU Idle: ${cpu_idle}%"
    fi
    
    # Memory
    if command -v free &> /dev/null; then
        local mem_used=$(free -h | grep Mem | awk '{print $3}')
        local mem_total=$(free -h | grep Mem | awk '{print $2}')
        echo -e "Memory: $mem_used / $mem_total"
    fi
    
    # Disk
    local disk_usage=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $5}')
    echo -e "Disk Usage: $disk_usage"
    
    # Docker (if in docker mode)
    if command -v docker &> /dev/null; then
        local docker_running=$(docker ps --format '{{.Names}}' | wc -l)
        echo -e "Docker Containers: $docker_running running"
    fi
}

check_docker_services() {
    echo ""
    echo "═══════════════════════════════════════════"
    echo "Docker Services"
    echo "═══════════════════════════════════════════"
    
    if ! command -v docker &> /dev/null; then
        echo "Docker not installed"
        return
    fi
    
    local services=(
        "redis"
        "postgres"
        "neo4j"
        "elasticsearch"
        "localai"
        "catalog"
        "extract-service"
        "graph-server"
        "deepagents-service"
    )
    
    for service in "${services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" --format '{{.Names}}' | grep -q "$service"; then
            local status=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "running")
            case "$status" in
                healthy)
                    echo -e "${GREEN}✓${NC} $service (healthy)"
                    ;;
                starting)
                    echo -e "${YELLOW}⊙${NC} $service (starting)"
                    ;;
                unhealthy)
                    echo -e "${RED}✗${NC} $service (unhealthy)"
                    ;;
                *)
                    echo -e "${GREEN}✓${NC} $service (running)"
                    ;;
            esac
        else
            echo -e "${RED}✗${NC} $service (not running)"
        fi
    done
}

################################################################################
# Main
################################################################################

main() {
    echo "═══════════════════════════════════════════"
    echo "aModels System Health Check"
    echo "═══════════════════════════════════════════"
    echo ""
    
    local failed=0
    local total=0
    
    # Check all services
    for service in "${!SERVICES[@]}"; do
        total=$((total + 1))
        if ! check_service "$service"; then
            failed=$((failed + 1))
        fi
    done
    
    # Detailed checks for critical services
    check_postgres_detail
    check_neo4j_detail
    check_elasticsearch_detail
    check_localai_detail
    
    # Docker services check
    check_docker_services
    
    # System resources
    check_system_resources
    
    # Summary
    echo ""
    echo "═══════════════════════════════════════════"
    echo "Summary"
    echo "═══════════════════════════════════════════"
    local healthy=$((total - failed))
    echo -e "Services: ${GREEN}$healthy healthy${NC}, ${RED}$failed failed${NC} (total: $total)"
    
    if [ $failed -eq 0 ]; then
        echo -e "\n${GREEN}✓ All services are healthy${NC}"
        exit 0
    else
        echo -e "\n${YELLOW}! Some services are not healthy${NC}"
        exit 1
    fi
}

main "$@"
