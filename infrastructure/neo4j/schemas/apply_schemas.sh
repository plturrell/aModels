#!/bin/bash
# Neo4j Schema Migration Script
# Executes all schema files in the correct order with error handling and logging

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# Configuration
# ============================================================================

NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"
SCHEMAS_DIR="${SCHEMAS_DIR:-$(dirname "$0")}"
LOG_FILE="${LOG_FILE:-${SCHEMAS_DIR}/schema_migration.log}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"

# ============================================================================
# Helper Functions
# ============================================================================

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

log_warn() {
    log "WARN" "$@"
}

log_success() {
    log "SUCCESS" "$@"
}

check_cypher_shell() {
    if ! command -v cypher-shell &> /dev/null; then
        log_error "cypher-shell not found. Please install Neo4j client tools."
        exit 1
    fi
}

test_connection() {
    log_info "Testing Neo4j connection..."
    if cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" \
        --format plain "RETURN 1 AS test;" > /dev/null 2>&1; then
        log_success "Connection successful"
        return 0
    else
        log_error "Failed to connect to Neo4j at $NEO4J_URI"
        return 1
    fi
}

execute_schema_file() {
    local file="$1"
    local phase="$2"
    
    if [ ! -f "$file" ]; then
        log_error "Schema file not found: $file"
        return 1
    fi
    
    log_info "Executing: $file (Phase: $phase)"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warn "DRY RUN: Would execute $file"
        return 0
    fi
    
    # Extract only UP MIGRATION section (lines between UP MIGRATION and DOWN MIGRATION)
    local temp_file=$(mktemp)
    awk '/^-- =+$/,/^-- =+$/ {
        if (/UP MIGRATION/) { in_up=1; next }
        if (/DOWN MIGRATION/) { exit }
        if (in_up) print
    }' "$file" > "$temp_file" || {
        # Fallback: if awk fails, use the whole file (for backward compatibility)
        cp "$file" "$temp_file"
    }
    
    # Remove comment-only lines and empty lines for cleaner execution
    sed -i '/^--/d; /^$/d' "$temp_file" || true
    
    if [ -s "$temp_file" ]; then
        if cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" \
            --format plain < "$temp_file" >> "$LOG_FILE" 2>&1; then
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

execute_down_migration() {
    local file="$1"
    
    if [ ! -f "$file" ]; then
        log_error "Schema file not found: $file"
        return 1
    fi
    
    log_info "Rolling back: $file"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warn "DRY RUN: Would rollback $file"
        return 0
    fi
    
    # Extract DOWN MIGRATION section
    local temp_file=$(mktemp)
    awk '/DOWN MIGRATION/,/^-- =+$/ {
        if (/DOWN MIGRATION/) { next }
        if (/^-- =+$/) { exit }
        print
    }' "$file" > "$temp_file"
    
    # Remove comment markers and empty lines
    sed -i 's/^-- //; /^$/d' "$temp_file" || true
    
    if [ -s "$temp_file" ]; then
        if cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" \
            --format plain < "$temp_file" >> "$LOG_FILE" 2>&1; then
            log_success "Successfully rolled back: $file"
            rm -f "$temp_file"
            return 0
        else
            log_error "Failed to rollback: $file"
            rm -f "$temp_file"
            return 1
        fi
    else
        log_warn "No rollback statements found in: $file"
        rm -f "$temp_file"
        return 0
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    local command="${1:-up}"
    
    # Initialize log file
    echo "=== Neo4j Schema Migration Log ===" > "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    echo "Neo4j URI: $NEO4J_URI" >> "$LOG_FILE"
    echo "User: $NEO4J_USER" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    log_info "Starting schema migration (command: $command)"
    
    check_cypher_shell
    
    if [ "$command" != "down" ]; then
        if ! test_connection; then
            exit 1
        fi
    fi
    
    case "$command" in
        up)
            execute_up_migrations
            ;;
        down)
            execute_down_migrations
            ;;
        *)
            echo "Usage: $0 [up|down]"
            echo ""
            echo "Commands:"
            echo "  up    - Apply all schema migrations (default)"
            echo "  down  - Rollback all schema migrations in reverse order"
            echo ""
            echo "Environment variables:"
            echo "  NEO4J_URI      - Neo4j connection URI (default: bolt://localhost:7687)"
            echo "  NEO4J_USER     - Neo4j username (default: neo4j)"
            echo "  NEO4J_PASSWORD - Neo4j password (default: password)"
            echo "  SCHEMAS_DIR    - Directory containing schema files (default: script directory)"
            echo "  LOG_FILE       - Log file path (default: schema_migration.log)"
            echo "  DRY_RUN        - Set to 'true' to preview without executing (default: false)"
            echo "  VERBOSE        - Set to 'true' for verbose output (default: false)"
            exit 1
            ;;
    esac
    
    log_success "Schema migration completed successfully"
}

execute_up_migrations() {
    local failed_files=()
    
    # Phase 1: Base schemas
    log_info "=== Phase 1: Base Schemas ==="
    for file in "$SCHEMAS_DIR"/base/*.cypher; do
        [ -f "$file" ] || continue
        if ! execute_schema_file "$file" "Base"; then
            failed_files+=("$file")
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
            failed_files+=("$file")
            log_error "Aborting due to failure in catalog schema"
            exit 1
        fi
    done
    
    # Regulatory domain
    log_info "--- Regulatory Domain ---"
    for file in "$SCHEMAS_DIR"/domain/regulatory/*.cypher; do
        [ -f "$file" ] || continue
        if ! execute_schema_file "$file" "Regulatory"; then
            failed_files+=("$file")
            log_error "Aborting due to failure in regulatory schema"
            exit 1
        fi
    done
    
    # Graph domain
    log_info "--- Graph Domain ---"
    for file in "$SCHEMAS_DIR"/domain/graph/*.cypher; do
        [ -f "$file" ] || continue
        if ! execute_schema_file "$file" "Graph"; then
            failed_files+=("$file")
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
            failed_files+=("$file")
        fi
    done
    
    if [ ${#failed_files[@]} -gt 0 ]; then
        log_error "Some migrations failed:"
        printf '  - %s\n' "${failed_files[@]}"
        return 1
    fi
}

execute_down_migrations() {
    # Execute in reverse order
    log_info "=== Rolling back migrations in reverse order ==="
    
    local files=()
    
    # Phase 3: Optimizations (reverse)
    while IFS= read -r file; do
        [ -f "$file" ] && files+=("$file")
    done < <(find "$SCHEMAS_DIR"/optimizations -name "*.cypher" -type f 2>/dev/null | sort -r)
    
    # Phase 2: Domain schemas (reverse order: graph -> regulatory -> catalog)
    while IFS= read -r file; do
        [ -f "$file" ] && files+=("$file")
    done < <(find "$SCHEMAS_DIR"/domain/graph -name "*.cypher" -type f 2>/dev/null | sort -r)
    
    while IFS= read -r file; do
        [ -f "$file" ] && files+=("$file")
    done < <(find "$SCHEMAS_DIR"/domain/regulatory -name "*.cypher" -type f 2>/dev/null | sort -r)
    
    while IFS= read -r file; do
        [ -f "$file" ] && files+=("$file")
    done < <(find "$SCHEMAS_DIR"/domain/catalog -name "*.cypher" -type f 2>/dev/null | sort -r)
    
    # Phase 1: Base schemas (reverse)
    while IFS= read -r file; do
        [ -f "$file" ] && files+=("$file")
    done < <(find "$SCHEMAS_DIR"/base -name "*.cypher" -type f 2>/dev/null | sort -r)
    
    # Execute all files in reverse order
    for file in "${files[@]}"; do
        execute_down_migration "$file" || log_warn "Rollback warning: $file"
    done
}

# Run main function
main "$@"

