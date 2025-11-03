#!/bin/bash
# Build script for AI Training Framework
# This script performs a full build including proto generation and testing

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo_info "Starting build process for AI Training Framework..."

# Check prerequisites
echo_info "Checking prerequisites..."

# Check Go
if ! command -v go &> /dev/null; then
    echo_error "Go is not installed"
    exit 1
fi
echo_info "Go version: $(go version)"

# Check protoc
if ! command -v protoc &> /dev/null; then
    echo_warn "protoc is not installed"
    echo_warn "Install from: https://grpc.io/docs/protoc-installation/"
    echo_warn "Skipping proto generation..."
    SKIP_PROTO=1
else
    echo_info "protoc version: $(protoc --version)"
fi

# Check protoc-gen-go
if ! command -v protoc-gen-go &> /dev/null; then
    if [ -z "$SKIP_PROTO" ]; then
        echo_warn "protoc-gen-go is not installed"
        echo_info "Installing proto tools..."
        make install-tools
    fi
fi

# Ensure Go bin is in PATH
export PATH=$PATH:$(go env GOPATH)/bin

# Step 1: Tidy dependencies
echo_info "Step 1/7: Tidying dependencies..."
make tidy

# Step 2: Verify module
echo_info "Step 2/7: Verifying module..."
make verify

# Step 3: Generate proto bindings (if tools available)
if [ -z "$SKIP_PROTO" ]; then
    echo_info "Step 3/7: Checking proto bindings..."
    make proto
    echo_info "Proto check complete"
else
    echo_warn "Step 3/7: Skipping proto generation (protoc not installed)"
fi

# Step 4: Build packages
echo_info "Step 4/7: Building packages..."
make build

# Step 5: Build binaries
echo_info "Step 5/7: Building binaries..."
make build-all

# Step 6: Run tests
echo_info "Step 6/7: Running tests..."
make test

# Step 7: Format check
echo_info "Step 7/7: Checking code formatting..."
make fmt

echo_info "Build completed successfully!"
echo_info ""
echo_info "Built binaries:"
echo_info "  - ./aibench"
echo_info "  - ./benchmark-server"
echo_info "  - ./arcagi_service"
echo_info ""
echo_info "To run:"
echo_info "  ./aibench list"
echo_info "  ./benchmark-server --help"
echo_info "  ./arcagi_service"
