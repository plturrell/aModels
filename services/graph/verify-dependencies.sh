#!/bin/bash
# Dependency Verification Script for Graph Service
# This script validates that all dependencies are correctly configured

set -e  # Exit on error

echo "========================================="
echo "Graph Service Dependency Verification"
echo "========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

function check_step() {
    local description=$1
    local command=$2
    
    echo -n "🔍 ${description}... "
    
    if eval "$command" &> /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((FAILED++))
        return 1
    fi
}

function info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Change to graph service directory
cd "$(dirname "$0")"

echo "📂 Working directory: $(pwd)"
echo ""

# 1. Check Go version
echo "=== Go Environment ==="
check_step "Go version >= 1.23" "go version | grep -E 'go1\.(2[3-9]|[3-9][0-9])'"
echo ""

# 2. Verify go.mod exists and is valid
echo "=== go.mod Validation ==="
check_step "go.mod file exists" "test -f go.mod"
check_step "go.mod is valid" "go mod verify"
check_step "All replace directives valid" "grep '^replace' go.mod | while read -r line; do
    path=\$(echo \$line | awk '{print \$NF}')
    if [[ \$path != http* ]]; then
        test -d \$path || test -f \$path/go.mod
    fi
done"
echo ""

# 3. Check critical dependencies
echo "=== Critical Dependencies ==="
check_step "Apache Arrow present" "go list -m github.com/apache/arrow-go/v18"
check_step "Neo4j driver present" "go list -m github.com/neo4j/neo4j-go-driver/v5"
check_step "gRPC present" "go list -m google.golang.org/grpc"
check_step "Internal catalog service" "go list -m github.com/plturrell/aModels/services/catalog"
check_step "Internal postgres service" "go list -m github.com/plturrell/aModels/services/postgres"
echo ""

# 4. Check replace directives
echo "=== Replace Directive Validation ==="
check_step "Self-reference replace exists" "grep -q 'replace github.com/plturrell/aModels/services/graph => .' go.mod"
check_step "Catalog replace exists" "grep -q 'replace github.com/plturrell/aModels/services/catalog => ../catalog' go.mod"
check_step "Extract replace exists" "grep -q 'replace github.com/plturrell/aModels/services/extract => ../extract' go.mod"
check_step "Postgres replace exists" "grep -q 'replace github.com/plturrell/aModels/services/postgres => ../postgres' go.mod"
echo ""

# 5. Verify relative paths
echo "=== Relative Path Validation ==="
check_step "Catalog service exists" "test -d ../catalog && test -f ../catalog/go.mod"
check_step "Extract service exists" "test -d ../extract && test -f ../extract/go.mod"
check_step "Postgres service exists" "test -d ../postgres && test -f ../postgres/go.mod"
check_step "Shared package exists" "test -d ../shared"
check_step "SAP HANA fork exists" "test -d ../../infrastructure/third_party/go-hdb"
echo ""

# 6. Try to download dependencies
echo "=== Dependency Download ==="
check_step "Download all dependencies" "go mod download"
echo ""

# 7. Check documentation
echo "=== Documentation ==="
check_step "DEPENDENCIES.md exists" "test -f DEPENDENCIES.md"
check_step "DEPENDENCY_FIX_SUMMARY.md exists" "test -f DEPENDENCY_FIX_SUMMARY.md"
check_step ".gitignore exists" "test -f .gitignore"
check_step "go.work in .gitignore" "grep -q 'go.work' .gitignore || grep -q 'go.work' ../../.gitignore"
echo ""

# 8. Optional: Check workspace
echo "=== Workspace Configuration (Optional) ==="
if [ -f "../../go.work.example" ]; then
    echo -e "${GREEN}✅ go.work.example exists${NC}"
    info "To enable workspace mode: cp ../../go.work.example ../../go.work"
else
    echo -e "${YELLOW}⚠️  go.work.example not found${NC}"
fi

if [ -f "../../go.work" ]; then
    echo -e "${GREEN}✅ go.work is active${NC}"
    check_step "Workspace includes graph service" "grep -q './services/graph' ../../go.work"
else
    echo -e "${YELLOW}⚠️  go.work not active (optional)${NC}"
    info "Workspace mode is optional - replace directives work without it"
fi
echo ""

# 9. Try a test build (optional)
echo "=== Build Test ==="
if check_step "Build graph-server binary" "go build -o /tmp/graph-server-test ./cmd/graph-server"; then
    rm -f /tmp/graph-server-test
    info "Binary can be built successfully"
fi
echo ""

# Summary
echo "========================================="
echo "📊 SUMMARY"
echo "========================================="
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All checks passed! Dependencies are properly configured.${NC}"
    echo ""
    echo "Next steps:"
    echo "  • Build the service: go build ./cmd/graph-server"
    echo "  • Run tests: go test ./..."
    echo "  • Start server: ./graph-server"
    echo "  • Read docs: cat DEPENDENCIES.md"
    exit 0
else
    echo -e "${RED}❌ Some checks failed. Review the errors above.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check DEPENDENCIES.md for detailed guidance"
    echo "  2. Verify you're in the mono-repo: /home/aModels"
    echo "  3. Ensure all sibling services exist (catalog, extract, postgres, shared)"
    echo "  4. Run: go mod tidy"
    echo "  5. Check: go mod verify"
    exit 1
fi
