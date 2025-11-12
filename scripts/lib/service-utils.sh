#!/bin/bash
################################################################################
# Service Utilities Library
# Shared functions for service management
################################################################################

# Service health check functions
check_http_health() {
    local url=$1
    local timeout=${2:-5}
    timeout "$timeout" curl -sf "$url" >/dev/null 2>&1
}

check_grpc_health() {
    local host=$1
    local port=$2
    if command -v grpc_health_probe &> /dev/null; then
        grpc_health_probe -addr="$host:$port" >/dev/null 2>&1
    else
        # Fallback to port check
        timeout 1 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null
    fi
}

check_postgres_health() {
    local host=${1:-localhost}
    local port=${2:-5432}
    local user=${3:-postgres}
    pg_isready -h "$host" -p "$port" -U "$user" >/dev/null 2>&1
}

check_redis_health() {
    local host=${1:-localhost}
    local port=${2:-6379}
    redis-cli -h "$host" -p "$port" ping >/dev/null 2>&1
}

check_neo4j_health() {
    local url=${1:-http://localhost:7474}
    curl -sf "$url" >/dev/null 2>&1
}

# Process management
get_service_pid() {
    local service_name=$1
    local pid_file="${PID_DIR}/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        cat "$pid_file"
    fi
}

is_service_running() {
    local service_name=$1
    local pid=$(get_service_pid "$service_name")
    
    if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

stop_service() {
    local service_name=$1
    local pid=$(get_service_pid "$service_name")
    
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        
        rm -f "${PID_DIR}/${service_name}.pid"
        return 0
    fi
    return 1
}

# Docker helpers
is_docker_service_running() {
    local service_name=$1
    docker ps --filter "name=$service_name" --filter "status=running" --format '{{.Names}}' | grep -q "$service_name"
}

docker_service_logs() {
    local service_name=$1
    local lines=${2:-100}
    docker logs --tail "$lines" "$service_name"
}

# Configuration helpers
load_env_file() {
    local env_file=$1
    if [ -f "$env_file" ]; then
        set -a
        source "$env_file"
        set +a
        return 0
    fi
    return 1
}

# Validation helpers
validate_go_service() {
    local service_path=$1
    if [ ! -f "${service_path}/main.go" ] && [ ! -f "${service_path}/go.mod" ]; then
        return 1
    fi
    return 0
}

validate_python_service() {
    local service_path=$1
    if [ ! -f "${service_path}/main.py" ] && [ ! -f "${service_path}/requirements.txt" ]; then
        return 1
    fi
    return 0
}

# Build helpers
build_go_service() {
    local service_path=$1
    local output_binary=$2
    
    cd "$service_path"
    go build -o "$output_binary" .
    return $?
}

install_python_deps() {
    local service_path=$1
    local venv_path=$2
    
    if [ -f "${service_path}/requirements.txt" ]; then
        if [ -n "$venv_path" ]; then
            source "${venv_path}/bin/activate"
        fi
        pip install -r "${service_path}/requirements.txt"
        return $?
    fi
    return 0
}
