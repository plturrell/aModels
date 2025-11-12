#!/bin/bash
# Consolidated Test Runner for aModels Project
# Combines best features from all test script variants
# 
# Features:
# - Auto-detects Docker vs host environment
# - Configures service URLs appropriately
# - Flexible error handling (continue on error by default)
# - Optional service health check
# - Comprehensive test tracking and reporting
#
# Usage:
#   ./run_all_tests.sh              # Run all tests with Docker URLs
#   ./run_all_tests.sh --host       # Use localhost URLs instead
#   ./run_all_tests.sh --check      # Run service check first
#   ./run_all_tests.sh --strict     # Exit on first failure (set -e)

set +e  # Don't exit on error by default - collect all test results

cd "$(dirname "$0")/../.."  # Go to project root

# Parse command line options
USE_DOCKER=true
RUN_SERVICE_CHECK=false
STRICT_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            USE_DOCKER=false
            shift
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --check)
            RUN_SERVICE_CHECK=true
            shift
            ;;
        --strict)
            STRICT_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host|--docker] [--check] [--strict]"
            exit 1
            ;;
    esac
done

# Auto-detect Docker environment if not explicitly set
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    echo "🐳 Detected Docker environment"
    USE_DOCKER=true
fi

# Set strict mode if requested
if [ "$STRICT_MODE" = true ]; then
    set -e
    echo "⚠️  Strict mode: will exit on first failure"
fi

echo "============================================================"
echo "Running All Tests - aModels Project"
echo "============================================================"
echo "Environment: $([ "$USE_DOCKER" = true ] && echo "Docker" || echo "Host")"
echo "Mode: $([ "$STRICT_MODE" = true ] && echo "Strict (exit on error)" || echo "Continue (collect all results)")"
echo ""

# Step 0: Optional service health check
if [ "$RUN_SERVICE_CHECK" = true ]; then
    echo "Step 0: Running service health check..."
    if ./scripts/testing/00_check_services.sh; then
        echo "✓ All services ready"
    else
        echo "⚠️  Some services not ready"
        if [ "$STRICT_MODE" = true ]; then
            echo "❌ Exiting due to service check failure (strict mode)"
            exit 1
        else
            echo "⚠️  Continuing anyway..."
        fi
    fi
    echo ""
fi

# Configure service URLs based on environment
if [ "$USE_DOCKER" = true ]; then
    # Docker network URLs (services accessible from Docker network)
    export LOCALAI_URL="http://localai:8080"
    export EXTRACT_SERVICE_URL="http://extract-service:8082"
    export TRAINING_SERVICE_URL="http://training-service:8080"
    export ORCHESTRATION_SERVICE_URL="http://orchestration-service:8080"
    export ANALYTICS_SERVICE_URL="http://analytics-service:8080"
    export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
    export REDIS_URL="redis://redis:6379/0"
    export NEO4J_URI="bolt://neo4j:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="password"
    
    echo "Using Docker network URLs:"
else
    # Host/localhost URLs (services accessible from host)
    export LOCALAI_URL="http://localhost:8081"
    export EXTRACT_SERVICE_URL="http://localhost:19080"
    export TRAINING_SERVICE_URL="http://localhost:8080"
    export ORCHESTRATION_SERVICE_URL="http://localhost:8080"
    export ANALYTICS_SERVICE_URL="http://localhost:8080"
    export POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/amodels"
    export REDIS_URL="redis://localhost:6379/0"
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="password"
    
    echo "Using localhost URLs:"
fi

echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  EXTRACT_SERVICE_URL: $EXTRACT_SERVICE_URL"
echo "  TRAINING_SERVICE_URL: $TRAINING_SERVICE_URL"
echo "  POSTGRES_DSN: $POSTGRES_DSN"
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
    
    if [ ! -f "$test_file" ]; then
        echo "⚠️  Test file not found: $test_file"
        return 1
    fi
    
    ((TOTAL_TESTS++))
    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "----------------------------------------"
    
    if bash "$test_file"; then
        echo "✅ PASSED: $test_name"
        ((PASSED_TESTS++))
        return 0
    else
        echo "❌ FAILED: $test_name"
        ((FAILED_TESTS++))
        FAILED_LIST+=("$test_name")
        return 1
    fi
}

# Run all test suites
echo ""
echo "============================================================"
echo "Starting Test Execution"
echo "============================================================"
echo ""

# Step 1: LocalAI Integration Tests
run_test "testing/01_test_localai.py" "LocalAI Integration"

# Step 2: Extract Service Tests
run_test "testing/02_test_extract_service.py" "Extract Service"

# Step 3: Training Service Tests
run_test "testing/03_test_training_service.py" "Training Service"

# Step 4: Orchestration Tests
run_test "testing/04_test_orchestration.py" "Orchestration Service"

# Step 5: Graph Integration Tests
run_test "testing/05_test_graph_integration.py" "Graph Integration"

# Step 6: End-to-End Workflow Tests
run_test "testing/06_test_e2e_workflow.py" "End-to-End Workflow"

# Additional Python tests if they exist
for test_file in testing/test_*.py; do
    if [ -f "$test_file" ]; then
        test_name=$(basename "$test_file" .py)
        # Skip if already run in the numbered tests
        if [[ ! "$test_name" =~ ^[0-9]+_ ]]; then
            run_test "$test_file" "$test_name"
        fi
    fi
done

# Summary Report
echo ""
echo "============================================================"
echo "Test Results Summary"
echo "============================================================"
echo "Total Tests:  $TOTAL_TESTS"
echo "Passed:       $PASSED_TESTS ($([ $TOTAL_TESTS -gt 0 ] && echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc || echo "0")%)"
echo "Failed:       $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_LIST[@]}"; do
        echo "  ❌ $test"
    done
    echo ""
    echo "❌ OVERALL: FAILED"
    exit 1
else
    echo "✅ OVERALL: ALL TESTS PASSED"
    exit 0
fi
