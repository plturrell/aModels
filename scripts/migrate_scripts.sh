#!/bin/bash
#
# Script Migration Tool for aModels Project
# Purpose: Reorganize and consolidate shell scripts into proper structure
# Usage: ./migrate_scripts.sh [--dry-run|--execute|--rollback]
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/aModels"
BACKUP_DIR="/tmp/scripts_backup_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/script_migration.log"

# Mode: dry-run, execute, or rollback
MODE="${1:-dry-run}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}ℹ${NC} $*" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✓${NC} $*" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}✗${NC} $*" | tee -a "$LOG_FILE"
}

banner() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$*${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Phase 1: Backup
backup_scripts() {
    banner "PHASE 1: Creating Backup"
    
    if [[ "$MODE" != "dry-run" ]]; then
        mkdir -p "$BACKUP_DIR"
        
        info "Backing up current scripts to: $BACKUP_DIR"
        tar -czf "${BACKUP_DIR}/scripts.tar.gz" \
            "${PROJECT_ROOT}/scripts" \
            "${PROJECT_ROOT}/testing" \
            "${PROJECT_ROOT}/tools" \
            "${PROJECT_ROOT}/services/start_all_services.sh" \
            "${PROJECT_ROOT}/services/telemetry-exporter/create_signavio_extract.sh" \
            2>/dev/null || true
        
        success "Backup created: ${BACKUP_DIR}/scripts.tar.gz"
    else
        info "DRY RUN: Would create backup at: $BACKUP_DIR"
    fi
}

# Phase 2: Create new directory structure
create_structure() {
    banner "PHASE 2: Creating New Directory Structure"
    
    local new_dir="${PROJECT_ROOT}/scripts/dev-tools"
    
    if [[ "$MODE" == "execute" ]]; then
        mkdir -p "$new_dir"
        success "Created: $new_dir"
    else
        info "DRY RUN: Would create: $new_dir"
    fi
}

# Phase 3: Move files
move_file() {
    local src="$1"
    local dest="$2"
    
    if [[ ! -f "$src" ]]; then
        warning "Source file not found: $src"
        return 1
    fi
    
    if [[ "$MODE" == "execute" ]]; then
        mkdir -p "$(dirname "$dest")"
        mv "$src" "$dest"
        success "Moved: $src -> $dest"
    else
        info "DRY RUN: Would move: $src -> $dest"
    fi
}

move_dev_tools() {
    banner "PHASE 3A: Moving Development Tools"
    
    local dev_tools_dest="${PROJECT_ROOT}/scripts/dev-tools"
    
    # From tools/helpers
    move_file \
        "${PROJECT_ROOT}/tools/helpers/fetch_kaggle_gemma.sh" \
        "${dev_tools_dest}/fetch_kaggle_gemma.sh"
    
    # From tools/scripts
    move_file \
        "${PROJECT_ROOT}/tools/scripts/build.sh" \
        "${dev_tools_dest}/build.sh"
    
    move_file \
        "${PROJECT_ROOT}/tools/scripts/cleanup_rt_archives.sh" \
        "${dev_tools_dest}/cleanup_rt_archives.sh"
    
    move_file \
        "${PROJECT_ROOT}/tools/scripts/run_rt_main_schedule.sh" \
        "${dev_tools_dest}/run_rt_main_schedule.sh"
    
    # From infrastructure
    move_file \
        "${PROJECT_ROOT}/infrastructure/docker/brev/sync-testing.sh" \
        "${dev_tools_dest}/sync-testing.sh"
}

move_system_scripts() {
    banner "PHASE 3B: Moving System Scripts"
    
    local system_dest="${PROJECT_ROOT}/scripts/system"
    
    move_file \
        "${PROJECT_ROOT}/services/start_all_services.sh" \
        "${system_dest}/start_all_services.sh"
}

move_signavio_scripts() {
    banner "PHASE 3C: Moving Signavio Scripts"
    
    local signavio_dest="${PROJECT_ROOT}/scripts/signavio"
    
    move_file \
        "${PROJECT_ROOT}/services/telemetry-exporter/create_signavio_extract.sh" \
        "${signavio_dest}/create_signavio_extract.sh"
}

move_testing_scripts() {
    banner "PHASE 3D: Moving Testing Scripts"
    
    local testing_dest="${PROJECT_ROOT}/scripts/testing"
    
    # Essential testing scripts
    local essential_tests=(
        "00_check_services.sh"
        "bootstrap_training_shell.sh"
        "setup_test_database.sh"
        "run_smoke_tests.sh"
        "test_localai_from_container.sh"
    )
    
    for script in "${essential_tests[@]}"; do
        if [[ -f "${PROJECT_ROOT}/testing/$script" ]]; then
            move_file \
                "${PROJECT_ROOT}/testing/$script" \
                "${testing_dest}/$script"
        fi
    done
    
    # Move performance benchmark to quality
    move_file \
        "${PROJECT_ROOT}/testing/performance_benchmark_runner.sh" \
        "${PROJECT_ROOT}/scripts/quality/performance_benchmark_runner.sh"
}

# Phase 4: Consolidate redundant scripts
consolidate_test_scripts() {
    banner "PHASE 4: Consolidating Redundant Test Scripts"
    
    local redundant_tests=(
        "run_all_tests_final.sh"
        "run_all_tests_fixed.sh"
        "run_all_tests_with_check.sh"
        "run_all_tests_with_step0.sh"
        "run_all_tests_working.sh"
        "run_tests_docker_network.sh"
        "run_tests_from_container.sh"
        "run_tests_from_docker.sh"
        "run_tests_in_container.sh"
        "run_tests_now.sh"
    )
    
    info "The following redundant scripts should be reviewed and possibly deleted:"
    for script in "${redundant_tests[@]}"; do
        if [[ -f "${PROJECT_ROOT}/testing/$script" ]]; then
            warning "  - ${PROJECT_ROOT}/testing/$script"
        fi
    done
    
    info "\nConsider consolidating into:"
    info "  - scripts/testing/run_all_tests.sh (consolidated test runner)"
    info "  - scripts/testing/run_tests_docker.sh (consolidated docker tests)"
}

# Phase 5: Update references
update_references() {
    banner "PHASE 5: Files That Need Reference Updates"
    
    local files_to_update=(
        "Makefile.services"
        "scripts/README.md"
        "scripts/REORGANIZATION.md"
        "docs/SERVICES_STARTUP.md"
        "QUICKSTART.md"
    )
    
    info "The following files need manual updates to reflect new paths:"
    for file in "${files_to_update[@]}"; do
        if [[ -f "${PROJECT_ROOT}/$file" ]]; then
            warning "  - $file"
        fi
    done
    
    echo ""
    info "Search for references with:"
    echo "  grep -r 'testing/' ${PROJECT_ROOT}/scripts/ ${PROJECT_ROOT}/Makefile* ${PROJECT_ROOT}/docs/"
    echo "  grep -r 'tools/scripts/' ${PROJECT_ROOT}/scripts/ ${PROJECT_ROOT}/Makefile* ${PROJECT_ROOT}/docs/"
    echo "  grep -r 'start_all_services.sh' ${PROJECT_ROOT}/scripts/ ${PROJECT_ROOT}/Makefile* ${PROJECT_ROOT}/docs/"
}

# Phase 6: Cleanup
cleanup() {
    banner "PHASE 6: Cleanup Empty Directories"
    
    local dirs_to_check=(
        "${PROJECT_ROOT}/testing"
        "${PROJECT_ROOT}/tools/scripts"
        "${PROJECT_ROOT}/tools/helpers"
    )
    
    for dir in "${dirs_to_check[@]}"; do
        if [[ -d "$dir" ]] && [[ -z "$(ls -A "$dir" 2>/dev/null)" ]]; then
            if [[ "$MODE" == "execute" ]]; then
                rmdir "$dir"
                success "Removed empty directory: $dir"
            else
                info "DRY RUN: Would remove empty directory: $dir"
            fi
        elif [[ -d "$dir" ]]; then
            warning "Directory not empty, manual review needed: $dir"
        fi
    done
}

# Generate migration report
generate_report() {
    banner "Migration Report"
    
    local report_file="${PROJECT_ROOT}/MIGRATION_REPORT_$(date +%Y%m%d).md"
    
    cat > "$report_file" << EOF
# Script Migration Report
Generated: $(date)
Mode: $MODE

## Summary

### Files Moved
- Dev Tools: 5 scripts moved to scripts/dev-tools/
- System: 1 script moved to scripts/system/
- Signavio: 1 script moved to scripts/signavio/
- Testing: 6 scripts moved to scripts/testing/

### Redundant Scripts Identified
Located in /testing/:
- run_all_tests_final.sh
- run_all_tests_fixed.sh
- run_all_tests_with_check.sh
- run_all_tests_with_step0.sh
- run_all_tests_working.sh
- run_tests_docker_network.sh
- run_tests_from_container.sh
- run_tests_from_docker.sh
- run_tests_in_container.sh
- run_tests_now.sh

**Recommendation:** Review and consolidate into 2-3 essential scripts.

### Manual Actions Required

1. **Update References**
   - Makefile.services
   - scripts/README.md
   - docs/SERVICES_STARTUP.md
   - QUICKSTART.md

2. **Review Redundant Scripts**
   - Analyze differences between run_all_tests*.sh variants
   - Consolidate docker test runners
   - Delete truly redundant copies

3. **Test After Migration**
   - Run: make -f Makefile.services health
   - Run: scripts/testing/test_all_services.sh
   - Verify all service startups

## Backup Location
$BACKUP_DIR

## Next Steps
1. Review this report
2. Test migrated scripts
3. Update documentation
4. Delete redundant scripts after verification
5. Remove empty directories

EOF
    
    success "Report generated: $report_file"
    cat "$report_file"
}

# Rollback function
rollback() {
    banner "ROLLBACK: Restoring from Backup"
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        error "Backup directory not found. Please specify backup location."
        echo "Usage: $0 --rollback <backup_dir>"
        exit 1
    fi
    
    local backup_tar="${BACKUP_DIR}/scripts.tar.gz"
    
    if [[ ! -f "$backup_tar" ]]; then
        error "Backup archive not found: $backup_tar"
        exit 1
    fi
    
    info "Restoring from: $backup_tar"
    tar -xzf "$backup_tar" -C /
    success "Rollback complete"
}

# Main execution
main() {
    log "Starting script migration - Mode: $MODE"
    
    case "$MODE" in
        --dry-run|dry-run)
            info "Running in DRY RUN mode - no changes will be made"
            backup_scripts
            create_structure
            move_dev_tools
            move_system_scripts
            move_signavio_scripts
            move_testing_scripts
            consolidate_test_scripts
            update_references
            cleanup
            generate_report
            ;;
        --execute|execute)
            warning "Running in EXECUTE mode - changes will be made"
            read -p "Continue? (yes/no): " confirm
            if [[ "$confirm" != "yes" ]]; then
                error "Migration cancelled"
                exit 1
            fi
            backup_scripts
            create_structure
            move_dev_tools
            move_system_scripts
            move_signavio_scripts
            move_testing_scripts
            consolidate_test_scripts
            update_references
            cleanup
            generate_report
            success "Migration complete!"
            ;;
        --rollback|rollback)
            rollback "${2:-$BACKUP_DIR}"
            ;;
        *)
            error "Invalid mode: $MODE"
            echo "Usage: $0 [--dry-run|--execute|--rollback]"
            exit 1
            ;;
    esac
}

main "$@"
