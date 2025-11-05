#!/bin/bash
# Comprehensive test runner for all LocalAI interaction points

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "LocalAI Integration Test Suite"
echo "=========================================="
echo ""

# Check if services are running
check_service() {
    local url=$1
    local name=$2
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $name is running"
        return 0
    else
        echo -e "${YELLOW}⚠️${NC} $name is not running (tests may be skipped)"
        return 1
    fi
}

echo "Checking services..."
check_service "http://localhost:8081/health" "LocalAI"
check_service "http://localhost:9004/healthz" "DeepAgents"
check_service "http://localhost:8080/health" "Graph Service"
check_service "http://localhost:8090/health" "Search-inference"
check_service "http://localhost:8082/health" "Extract Service"
check_service "http://localhost:8000/health" "Gateway"
check_service "http://localhost:9090/health" "Transformers Service"
echo ""

# Run Python tests
echo "=========================================="
echo "Running Python Integration Tests"
echo "=========================================="
if command -v python3 &> /dev/null; then
    python3 testing/test_localai_integration_suite.py
    PYTHON_EXIT=$?
else
    echo -e "${YELLOW}⚠️${NC} Python3 not found, skipping Python tests"
    PYTHON_EXIT=0
fi
echo ""

# Run Go tests
echo "=========================================="
echo "Running Go Integration Tests"
echo "=========================================="
if command -v go &> /dev/null; then
    cd testing
    go test -v -timeout 60s ./test_embedding_models.go ./test_service_integrations.go ./test_deepseek_ocr.go
    GO_EXIT=$?
    cd ..
else
    echo -e "${YELLOW}⚠️${NC} Go not found, skipping Go tests"
    GO_EXIT=0
fi
echo ""

# Run DeepAgents specific test
echo "=========================================="
echo "Running DeepAgents → LocalAI Test"
echo "=========================================="
if [ -f scripts/test_deepagents_localai.py ]; then
    python3 scripts/test_deepagents_localai.py
    DEEPAGENTS_EXIT=$?
else
    echo -e "${YELLOW}⚠️${NC} DeepAgents test script not found"
    DEEPAGENTS_EXIT=0
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="

if [ $PYTHON_EXIT -eq 0 ] && [ $GO_EXIT -eq 0 ] && [ $DEEPAGENTS_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    echo "Python tests: $([ $PYTHON_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
    echo "Go tests: $([ $GO_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
    echo "DeepAgents tests: $([ $DEEPAGENTS_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
    exit 1
fi

