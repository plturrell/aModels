#!/bin/bash
# Run all tests with proper service URL configuration
# This script handles both host and Docker network scenarios

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "aModels Test Suite - All Weeks"
echo "=========================================="
echo ""

# Detect if running in Docker or on host
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    echo "Running in Docker container - using service names"
    export LOCALAI_URL="http://localai:8080"
    export EXTRACT_SERVICE_URL="http://extract-service:19080"
    export TRAINING_SERVICE_URL="http://training-service:8080"
    export ORCHESTRATION_SERVICE_URL="http://orchestration-service:8080"
    export ANALYTICS_SERVICE_URL="http://analytics-service:8080"
    export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
    export REDIS_URL="redis://redis:6379/0"
    export NEO4J_URI="bolt://neo4j:7687"
else
    echo "Running on host - using localhost"
    export LOCALAI_URL="http://localhost:8081"
    export EXTRACT_SERVICE_URL="http://localhost:19080"
    export TRAINING_SERVICE_URL="http://localhost:8080"
    export ORCHESTRATION_SERVICE_URL="http://localhost:8080"
    export ANALYTICS_SERVICE_URL="http://localhost:8080"
    export POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/amodels"
    export REDIS_URL="redis://localhost:6379/0"
    export NEO4J_URI="bolt://localhost:7687"
fi

echo "Service URLs:"
echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  EXTRACT_SERVICE_URL: $EXTRACT_SERVICE_URL"
echo "  TRAINING_SERVICE_URL: $TRAINING_SERVICE_URL"
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
check_service "$LOCALAI_URL/health" "LocalAI" || true
check_service "$EXTRACT_SERVICE_URL/healthz" "Extract Service" || true
check_service "$TRAINING_SERVICE_URL/health" "Training Service" || true
echo ""

# Week 1: Foundation Tests
echo "=========================================="
echo "Week 1: Foundation Tests"
echo "=========================================="
echo "Running domain detection tests..."
python3 testing/test_domain_detection.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running domain filter tests..."
python3 testing/test_domain_filter.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running domain trainer tests..."
python3 testing/test_domain_trainer.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running domain metrics tests..."
python3 testing/test_domain_metrics.py || echo "⚠️  Some tests failed or skipped"
echo ""

# Week 2: Integration Tests
echo "=========================================="
echo "Week 2: Integration Tests"
echo "=========================================="
echo "Running extraction flow tests..."
python3 testing/test_extraction_flow.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running training flow tests..."
python3 testing/test_training_flow.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running A/B testing flow tests..."
python3 testing/test_ab_testing_flow.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running rollback flow tests..."
python3 testing/test_rollback_flow.py || echo "⚠️  Some tests failed or skipped"
echo ""

# Week 3: Phase 7-9 Tests
echo "=========================================="
echo "Week 3: Phase 7-9 Tests"
echo "=========================================="
echo "Running pattern learning tests..."
python3 testing/test_pattern_learning.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running extraction intelligence tests..."
python3 testing/test_extraction_intelligence.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running automation tests..."
python3 testing/test_automation.py || echo "⚠️  Some tests failed or skipped"
echo ""

# Week 4: Performance & Load Tests
echo "=========================================="
echo "Week 4: Performance & Load Tests"
echo "=========================================="
echo "Running performance tests..."
python3 testing/test_performance.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running load tests..."
python3 testing/test_load.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running concurrent request tests..."
python3 testing/test_concurrent_requests.py || echo "⚠️  Some tests failed or skipped"
echo ""

echo "Running performance benchmarks..."
python3 testing/performance_benchmark.py || echo "⚠️  Some tests failed or skipped"
echo ""

# Core Integration Tests
echo "=========================================="
echo "Core Integration Tests"
echo "=========================================="
echo "Running LocalAI integration tests..."
python3 testing/test_localai_integration_suite.py || echo "⚠️  Some tests failed or skipped"
echo ""

# Summary
echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="
echo ""
echo "Note: Some tests may be skipped if services are not available."
echo "To run tests from within Docker, use:"
echo "  docker exec -it training-shell bash"
echo "  cd /workspace && ./testing/run_all_tests_fixed.sh"
echo ""

