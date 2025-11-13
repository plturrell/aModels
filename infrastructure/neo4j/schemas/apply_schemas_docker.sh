#!/bin/bash
# Neo4j Schema Migration Script (Docker version)
# Executes all schema files using docker exec

set -euo pipefail

# Configuration
NEO4J_CONTAINER="${NEO4J_CONTAINER:-neo4j}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-amodels123}"
SCHEMAS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCHEMAS_DIR}/schema_migration.log"

# Helper functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$@"
}

log_error() {
    log "ERROR" "$@" >&2
}

log_success() {
    log "SUCCESS" "$@"
}

log_warn() {
    log "WARN" "$@"
}

execute_schema_file() {
    local file="$1"
    local phase="$2"
    
    if [ ! -f "$file" ]; then
        log_error "Schema file not found: $file"
        return 1
    fi
    
    log_info "Executing: $file (Phase: $phase)"
    
    # Extract only UP MIGRATION section
    local temp_file=$(mktemp)
    awk '/^-- =+$/,/^-- =+$/ {
        if (/UP MIGRATION/) { in_up=1; next }
        if (/DOWN MIGRATION/) { exit }
        if (in_up) print
    }' "$file" > "$temp_file" || {
        # Fallback: if awk fails, use the whole file
        cp "$file" "$temp_file"
    }
    
    # Remove comment-only lines and empty lines
    sed -i '/^--/d; /^$/d' "$temp_file" || true
    
    if [ -s "$temp_file" ]; then
        # Copy temp file to container and execute
        docker cp "$temp_file" "${NEO4J_CONTAINER}:/tmp/migration.cypher" > /dev/null 2>&1
        if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
            -a bolt://localhost:7687 --format plain < "$temp_file" >> "$LOG_FILE" 2>&1; then
            log_success "Successfully executed: $file"
            rm -f "$temp_file"
            return 0
        else
            log_error "Failed to execute: $file"
            rm -f "$temp_file"
            return 1
        fi
    else
        log_warn "No migration statements found in: $file"
        rm -f "$temp_file"
        return 0
    fi
}

# Initialize log
echo "=== Neo4j Schema Migration Log (Docker) ===" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "Container: $NEO4J_CONTAINER" >> "$LOG_FILE"
echo "User: $NEO4J_USER" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

log_info "Starting schema migration using Docker container: $NEO4J_CONTAINER"

# Test connection
log_info "Testing Neo4j connection..."
if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
    -a bolt://localhost:7687 --format plain "RETURN 1 AS test;" > /dev/null 2>&1; then
    log_success "Connection successful"
else
    log_error "Failed to connect to Neo4j"
    exit 1
fi

# Phase 1: Base schemas
log_info "=== Phase 1: Base Schemas ==="
for file in "$SCHEMAS_DIR"/base/*.cypher; do
    [ -f "$file" ] || continue
    if ! execute_schema_file "$file" "Base"; then
        log_error "Aborting due to failure in base schema"
        exit 1
    fi
done

# Phase 2: Domain schemas
log_info "=== Phase 2: Domain Schemas ==="

# Catalog domain
log_info "--- Catalog Domain ---"
for file in "$SCHEMAS_DIR"/domain/catalog/*.cypher; do
    [ -f "$file" ] || continue
    if ! execute_schema_file "$file" "Catalog"; then
        log_error "Aborting due to failure in catalog schema"
        exit 1
    fi
done

# Regulatory domain
log_info "--- Regulatory Domain ---"
for file in "$SCHEMAS_DIR"/domain/regulatory/*.cypher; do
    [ -f "$file" ] || continue
    if ! execute_schema_file "$file" "Regulatory"; then
        log_error "Aborting due to failure in regulatory schema"
        exit 1
    fi
done

# Graph domain
log_info "--- Graph Domain ---"
for file in "$SCHEMAS_DIR"/domain/graph/*.cypher; do
    [ -f "$file" ] || continue
    if ! execute_schema_file "$file" "Graph"; then
        log_error "Aborting due to failure in graph schema"
        exit 1
    fi
done

# Phase 3: Optimizations (optional)
log_info "=== Phase 3: Optimizations (Optional) ==="
for file in "$SCHEMAS_DIR"/optimizations/*.cypher; do
    [ -f "$file" ] || continue
    if ! execute_schema_file "$file" "Optimization"; then
        log_warn "Optimization failed but continuing: $file"
    fi
done

log_success "Schema migration completed successfully"
echo ""
echo "Migration log saved to: $LOG_FILE"

