#!/bin/bash
# Run all tests from within a Docker container where services are accessible
# This script creates a temporary container on the same network as LocalAI

set +e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Running All Tests from Docker Container"
echo "============================================================"
echo ""

# Get the network name from LocalAI container
NETWORK_NAME=$(docker inspect localai --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null | head -1)

if [ -z "$NETWORK_NAME" ]; then
    echo "❌ Cannot find Docker network. Is LocalAI running?"
    exit 1
fi

echo "Using Docker network: $NETWORK_NAME"
echo ""

# Set environment variables for tests
export LOCALAI_URL="http://localai:8080"
export EXTRACT_SERVICE_URL="http://extract-service:19080"
export TRAINING_SERVICE_URL="http://training-service:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
export REDIS_URL="redis://redis:6379/0"
export NEO4J_URI="bolt://neo4j:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

echo "Environment:"
echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  EXTRACT_SERVICE_URL: $EXTRACT_SERVICE_URL"
echo ""

# Create and run test container
echo "Creating test container..."
docker run --rm \
    --network "$NETWORK_NAME" \
    --name aModels-test-runner-$$ \
    -v "$(pwd):/workspace" \
    -w /workspace \
    -e LOCALAI_URL="$LOCALAI_URL" \
    -e EXTRACT_SERVICE_URL="$EXTRACT_SERVICE_URL" \
    -e TRAINING_SERVICE_URL="$TRAINING_SERVICE_URL" \
    -e POSTGRES_DSN="$POSTGRES_DSN" \
    -e REDIS_URL="$REDIS_URL" \
    -e NEO4J_URI="$NEO4J_URI" \
    -e NEO4J_USER="$NEO4J_USER" \
    -e NEO4J_PASSWORD="$NEO4J_PASSWORD" \
    python:3.10-slim bash -c "
        echo 'Installing dependencies...'
        pip install -q httpx psycopg2-binary redis neo4j > /dev/null 2>&1
        
        echo ''
        echo '============================================================'
        echo 'Running All Tests'
        echo '============================================================'
        echo ''
        
        # Week 1 Tests
        echo '=== Week 1: Foundation Tests ==='
        python3 testing/test_domain_detection.py 2>&1 | tail -15
        echo ''
        python3 testing/test_domain_filter.py 2>&1 | tail -15
        echo ''
        python3 testing/test_domain_trainer.py 2>&1 | tail -15
        echo ''
        python3 testing/test_domain_metrics.py 2>&1 | tail -15
        echo ''
        
        # Week 2 Tests
        echo '=== Week 2: Integration Tests ==='
        python3 testing/test_extraction_flow.py 2>&1 | tail -15
        echo ''
        python3 testing/test_training_flow.py 2>&1 | tail -15
        echo ''
        python3 testing/test_ab_testing_flow.py 2>&1 | tail -15
        echo ''
        python3 testing/test_rollback_flow.py 2>&1 | tail -15
        echo ''
        
        # Week 3 Tests
        echo '=== Week 3: Pattern Learning & Intelligence ==='
        python3 testing/test_pattern_learning.py 2>&1 | tail -15
        echo ''
        python3 testing/test_extraction_intelligence.py 2>&1 | tail -15
        echo ''
        python3 testing/test_automation.py 2>&1 | tail -15
        echo ''
        
        # Week 4 Tests
        echo '=== Week 4: Performance & Load Tests ==='
        python3 testing/test_performance.py 2>&1 | tail -15
        echo ''
        python3 testing/test_load.py 2>&1 | tail -15
        echo ''
        python3 testing/test_concurrent_requests.py 2>&1 | tail -15
        echo ''
        python3 testing/performance_benchmark.py 2>&1 | tail -15
        echo ''
        
        # Integration Suite
        echo '=== Integration Suite ==='
        python3 testing/test_localai_integration_suite.py 2>&1 | tail -15
        echo ''
        
        echo '============================================================'
        echo 'Test Execution Complete'
        echo '============================================================'
    " 2>&1

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Tests completed"
else
    echo "⚠️  Some tests may have failed (check output above)"
fi

exit $EXIT_CODE

