#!/bin/bash
################################################################################
# Docker Compose Manager
# Simplified Docker Compose operations with profiles and service groups
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml"

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Service groups
readonly INFRASTRUCTURE="redis postgres neo4j elasticsearch"
readonly CORE="localai transformers config-sync localai-compat"
readonly APP="catalog extract graph search-inference deepagents"
readonly OPTIONAL="runtime orchestration training-service dms"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

################################################################################
# Docker Compose Operations
################################################################################

start_services() {
    local services="$*"
    
    log_info "Starting services: $services"
    docker-compose -f "$COMPOSE_FILE" up -d $services
    
    log_success "Services started"
}

stop_services() {
    local services="$*"
    
    if [ -z "$services" ]; then
        log_info "Stopping all services"
        docker-compose -f "$COMPOSE_FILE" down
    else
        log_info "Stopping services: $services"
        docker-compose -f "$COMPOSE_FILE" stop $services
    fi
    
    log_success "Services stopped"
}

restart_services() {
    local services="$*"
    
    log_info "Restarting services: $services"
    docker-compose -f "$COMPOSE_FILE" restart $services
    
    log_success "Services restarted"
}

show_logs() {
    local services="$*"
    local follow="${FOLLOW:-false}"
    
    if [ "$follow" = "true" ]; then
        docker-compose -f "$COMPOSE_FILE" logs -f $services
    else
        docker-compose -f "$COMPOSE_FILE" logs --tail=100 $services
    fi
}

show_status() {
    log_info "Service Status:"
    echo ""
    docker-compose -f "$COMPOSE_FILE" ps
}

build_services() {
    local services="$*"
    
    if [ -z "$services" ]; then
        log_info "Building all services"
        docker-compose -f "$COMPOSE_FILE" build
    else
        log_info "Building services: $services"
        docker-compose -f "$COMPOSE_FILE" build $services
    fi
    
    log_success "Build complete"
}

pull_images() {
    log_info "Pulling latest images"
    docker-compose -f "$COMPOSE_FILE" pull
    log_success "Images updated"
}

clean_volumes() {
    log_info "Cleaning volumes (this will delete all data!)"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        docker-compose -f "$COMPOSE_FILE" down -v
        log_success "Volumes cleaned"
    else
        log_info "Cancelled"
    fi
}

exec_service() {
    local service=$1
    shift
    local cmd="$*"
    
    if [ -z "$cmd" ]; then
        cmd="/bin/bash"
    fi
    
    log_info "Executing in $service: $cmd"
    docker-compose -f "$COMPOSE_FILE" exec $service $cmd
}

################################################################################
# Service Group Operations
################################################################################

start_infrastructure() {
    log_info "Starting infrastructure services"
    start_services $INFRASTRUCTURE
    
    # Wait for critical services
    sleep 10
    log_success "Infrastructure ready"
}

start_core() {
    log_info "Starting core services"
    start_services $CORE
    
    sleep 15
    log_success "Core services ready"
}

start_app() {
    log_info "Starting application services"
    start_services $APP
    
    sleep 10
    log_success "Application services ready"
}

start_full_stack() {
    start_infrastructure
    start_core
    start_app
    
    log_success "Full stack is running"
    show_status
}

################################################################################
# CLI
################################################################################

show_help() {
    cat << EOF
Docker Compose Manager for aModels

Usage: $0 COMMAND [OPTIONS]

Commands:
    start [services]        Start services (or all if none specified)
    stop [services]         Stop services (or all if none specified)
    restart [services]      Restart services
    logs [services]         Show logs (use FOLLOW=true for live logs)
    status                  Show service status
    build [services]        Build service images
    pull                    Pull latest images
    clean                   Clean volumes (WARNING: deletes data)
    exec SERVICE [cmd]      Execute command in service container
    
Service Groups:
    infrastructure          Start infrastructure (redis, postgres, neo4j, elasticsearch)
    core                    Start core services (localai, transformers)
    app                     Start application services (catalog, extract, graph, etc.)
    full                    Start full stack in correct order

Examples:
    $0 start infrastructure
    $0 start localai catalog extract
    $0 stop
    $0 logs catalog
    FOLLOW=true $0 logs localai
    $0 exec catalog /bin/bash
    $0 status

EOF
}

main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi
    
    local command=$1
    shift
    
    case "$command" in
        start)
            if [ $# -eq 0 ]; then
                start_full_stack
            elif [ "$1" = "infrastructure" ]; then
                start_infrastructure
            elif [ "$1" = "core" ]; then
                start_core
            elif [ "$1" = "app" ]; then
                start_app
            elif [ "$1" = "full" ]; then
                start_full_stack
            else
                start_services "$@"
            fi
            ;;
        stop)
            stop_services "$@"
            ;;
        restart)
            restart_services "$@"
            ;;
        logs)
            show_logs "$@"
            ;;
        status|ps)
            show_status
            ;;
        build)
            build_services "$@"
            ;;
        pull)
            pull_images
            ;;
        clean)
            clean_volumes
            ;;
        exec)
            exec_service "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
