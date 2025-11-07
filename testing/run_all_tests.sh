#!/bin/bash
# Run all tests with proper Docker network URLs
# This script ensures tests use the correct service URLs

set +e  # Don't exit on error - collect all test results

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Running All Tests"
echo "============================================================"
echo ""

# Step 0: Verify services first
echo "Step 0: Verifying services..."
if ! ./testing/00_check_services.sh > /dev/null 2>&1; then
    echo "⚠️  Some services not ready, but continuing..."
fi
echo ""

# Set Docker network URLs (services accessible from Docker network)
export LOCALAI_URL="http://localai:8080"
export EXTRACT_SERVICE_URL="http://extract-service:8082"
export TRAINING_SERVICE_URL="http://training-service:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
export REDIS_URL="redis://redis:6379/0"
export NEO4J_URI="bolt://neo4j:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

echo "Using Docker network URLs:"
echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  EXTRACT_SERVICE_URL: $EXTRACT_SERVICE_URL"
echo "  TRAINING_SERVICE_URL: $TRAINING_SERVICE_URL"
echo ""

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
FAILED_LIST=()

# Function to run a test and track results
run_test() {
    local test_file=$1
    local test_name=$2
    
    if [ ! -f "testing/$test_file" ]; then
        echo "⚠️  Test file not found: testing/$test_file"
        return 1
    fi
    
    echo "============================================================"
    echo "Running: $test_name"
    echo "File: $test_file"
    echo "============================================================"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if python3 "testing/$test_file" 2>&1; then
        echo ""
        echo "✅ $test_name: PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo ""
        echo "❌ $test_name: FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_LIST+=("$test_name")
        return 1
    fi
}

# Week 1: Foundation Tests
echo "============================================================"
echo "WEEK 1: Foundation Tests"
echo "============================================================"
echo ""

run_test "test_domain_detection.py" "Domain Detection"
run_test "test_domain_filter.py" "Domain Filter"
run_test "test_domain_trainer.py" "Domain Trainer"
run_test "test_domain_metrics.py" "Domain Metrics"

# Week 2: Integration Tests
echo ""
echo "============================================================"
echo "WEEK 2: Integration Tests"
echo "============================================================"
echo ""

run_test "test_extraction_flow.py" "Extraction Flow"
run_test "test_training_flow.py" "Training Flow"
run_test "test_ab_testing_flow.py" "A/B Testing Flow"
run_test "test_rollback_flow.py" "Rollback Flow"

# Week 3: Pattern Learning & Intelligence
echo ""
echo "============================================================"
echo "WEEK 3: Pattern Learning & Intelligence"
echo "============================================================"
echo ""

run_test "test_pattern_learning.py" "Pattern Learning"
run_test "test_extraction_intelligence.py" "Extraction Intelligence"
run_test "test_automation.py" "Automation"

# Week 4: Performance & Load Tests
echo ""
echo "============================================================"
echo "WEEK 4: Performance & Load Tests"
echo "============================================================"
echo ""

run_test "test_performance.py" "Performance Tests"
run_test "test_load.py" "Load Tests"
run_test "test_concurrent_requests.py" "Concurrent Requests"
run_test "performance_benchmark.py" "Performance Benchmark"

# Integration Suite
echo ""
echo "============================================================"
echo "Integration Suite"
echo "============================================================"
echo ""

run_test "test_localai_integration_suite.py" "LocalAI Integration Suite"

# Final Summary
echo ""
echo "============================================================"
echo "Test Execution Summary"
echo "============================================================"
echo ""
echo "Total Tests Run: $TOTAL_TESTS"
echo "✅ Passed: $PASSED_TESTS"
echo "❌ Failed: $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_LIST[@]}"; do
        echo "  ❌ $test"
    done
    echo ""
    echo "============================================================"
    echo "❌ SOME TESTS FAILED"
    echo "============================================================"
    exit 1
else
    echo "============================================================"
    echo "✅ ALL TESTS PASSED"
    echo "============================================================"
    exit 0
fi
