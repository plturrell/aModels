#!/bin/bash
# Week 4: Performance Benchmark Runner
# Runs all performance and load tests and generates a report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Performance Benchmark Runner - Week 4"
echo "=========================================="
echo ""

# Check prerequisites
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌${NC} Python3 not found"
    exit 1
fi

# Check services
check_service() {
    local url=$1
    local name=$2
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $name is running"
        return 0
    else
        echo -e "${YELLOW}⚠️${NC} $name is not running"
        return 1
    fi
}

echo "Checking services..."
check_service "http://localhost:8081/health" "LocalAI" || true
check_service "http://localhost:19080/healthz" "Extract Service" || true
check_service "http://localhost:8080/health" "Training Service" || true
echo ""

# Create results directory
RESULTS_DIR="testing/performance_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$RESULTS_DIR/performance_report_${TIMESTAMP}.txt"

echo "Results will be saved to: $REPORT_FILE"
echo ""

# Run performance tests
echo "=========================================="
echo "Running Performance Tests"
echo "=========================================="
python3 testing/test_performance.py 2>&1 | tee -a "$REPORT_FILE"
PERF_EXIT=$?
echo ""

# Run load tests
echo "=========================================="
echo "Running Load Tests"
echo "=========================================="
python3 testing/test_load.py 2>&1 | tee -a "$REPORT_FILE"
LOAD_EXIT=$?
echo ""

# Run concurrent domain tests
echo "=========================================="
echo "Running Concurrent Domain Tests"
echo "=========================================="
python3 testing/test_concurrent_domains.py 2>&1 | tee -a "$REPORT_FILE"
CONCURRENT_EXIT=$?
echo ""

# Run large graph tests
echo "=========================================="
echo "Running Large Graph Tests"
echo "=========================================="
python3 testing/test_large_graphs.py 2>&1 | tee -a "$REPORT_FILE"
LARGE_GRAPH_EXIT=$?
echo ""

# Summary
echo "=========================================="
echo "Performance Benchmark Summary"
echo "=========================================="
echo ""

if [ $PERF_EXIT -eq 0 ] && [ $LOAD_EXIT -eq 0 ] && [ $CONCURRENT_EXIT -eq 0 ] && [ $LARGE_GRAPH_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ All performance tests passed!${NC}"
    echo ""
    echo "Performance Report: $REPORT_FILE"
    exit 0
else
    echo -e "${RED}❌ Some performance tests failed${NC}"
    echo "Performance Report: $REPORT_FILE"
    echo ""
    echo "Test Results:"
    echo "  Performance tests: $([ $PERF_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
    echo "  Load tests: $([ $LOAD_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
    echo "  Concurrent domain tests: $([ $CONCURRENT_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
    echo "  Large graph tests: $([ $LARGE_GRAPH_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
    exit 1
fi

