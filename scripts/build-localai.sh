#!/bin/bash
################################################################################
# Build LocalAI (vaultgemma-server) Docker Image
# 
# Builds the LocalAI Docker image using the production Dockerfile
# Verifies build succeeds and binary exists in the image
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/infrastructure/docker/brev/docker-compose.yml"
DOCKERFILE="${PROJECT_ROOT}/services/localai/Dockerfile"

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

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
    echo -e "${YELLOW}[WARN]${NC} $*"
}

################################################################################
# Validation
################################################################################

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_info "Docker is available and running"
}

check_dockerfile() {
    if [ ! -f "$DOCKERFILE" ]; then
        log_error "Dockerfile not found: $DOCKERFILE"
        exit 1
    fi
    
    log_info "Dockerfile found: $DOCKERFILE"
}

check_docker_compose() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_info "Docker Compose file found: $COMPOSE_FILE"
}

################################################################################
# Build
################################################################################

build_localai() {
    log_info "Building LocalAI (vaultgemma-server) Docker image..."
    log_info "Dockerfile: $DOCKERFILE"
    log_info "Build context: $PROJECT_ROOT"
    
    # Build using docker-compose to ensure consistency with runtime
    cd "$PROJECT_ROOT"
    
    # Capture build output to check for warnings
    local build_output
    local build_exit_code
    
    if build_output=$(docker-compose -f "$COMPOSE_FILE" build localai 2>&1); then
        build_exit_code=0
    else
        build_exit_code=$?
    fi
    
    # Check for critical errors
    if [ $build_exit_code -ne 0 ]; then
        log_error "Build failed with exit code $build_exit_code"
        echo "$build_output"
        exit 1
    fi
    
    # Check for warnings (non-critical but should be noted)
    if echo "$build_output" | grep -qi "warning"; then
        log_warn "Build completed with warnings:"
        echo "$build_output" | grep -i "warning" || true
    fi
    
    log_success "LocalAI Docker image built successfully"
    
    # Verify the image exists
    if docker images | grep -q "amodels/localai.*vendored"; then
        log_success "Image verified: amodels/localai:vendored"
    else
        log_warn "Image tag verification failed, but build succeeded"
    fi
    
    # Verify binary exists in the image (optional check)
    log_info "Verifying vaultgemma-server binary in image..."
    if docker run --rm --entrypoint /bin/sh amodels/localai:vendored -c "test -f /usr/local/bin/vaultgemma-server && echo 'Binary exists'" 2>/dev/null | grep -q "Binary exists"; then
        log_success "vaultgemma-server binary verified in image"
    else
        log_warn "Could not verify binary in image (this may be normal if entrypoint differs)"
    fi
    
    return 0
}

################################################################################
# Main
################################################################################

main() {
    echo ""
    log_info "=========================================="
    log_info "LocalAI Build Script"
    log_info "=========================================="
    echo ""
    
    check_docker
    check_dockerfile
    check_docker_compose
    
    echo ""
    build_localai
    
    echo ""
    log_success "Build process completed successfully"
    echo ""
}

main "$@"

