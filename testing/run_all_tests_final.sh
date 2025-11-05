#!/bin/bash
# Run all tests from a Docker container with proper network and mounts
# This is the final working version that runs all tests

set +e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Running All Tests"
echo "============================================================"
echo ""

# Get Docker network
NETWORK=$(docker inspect localai --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null | head -1)

if [ -z "$NETWORK" ]; then
    echo "❌ LocalAI container not found. Is it running?"
    exit 1
fi

echo "Using Docker network: $NETWORK"
echo ""

# Run tests in container
docker run --rm \
    --network "$NETWORK" \
    -v "$(pwd):/home/aModels" \
    -w /home/aModels \
    -e LOCALAI_URL="http://localai:8080" \
    -e EXTRACT_SERVICE_URL="http://extract-service:19080" \
    -e TRAINING_SERVICE_URL="http://training-service:8080" \
    -e POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels" \
    -e REDIS_URL="redis://redis:6379/0" \
    -e NEO4J_URI="bolt://neo4j:7687" \
    -e NEO4J_USER="neo4j" \
    -e NEO4J_PASSWORD="password" \
    python:3.10-slim bash -c "
        echo 'Installing dependencies...'
        pip install -q httpx psycopg2-binary redis neo4j > /dev/null 2>&1
        echo '✅ Dependencies installed'
        echo ''
        
        echo 'Testing LocalAI connectivity...'
        python3 -c 'import httpx; r=httpx.get(\"http://localai:8080/readyz\", timeout=5); print(f\"✅ LocalAI accessible: {r.status_code}\")' 2>&1
        echo ''
        
        # Track results
        TOTAL=0
        PASSED=0
        FAILED=0
        
        # Week 1
        echo '=== Week 1: Foundation Tests ==='
        for test in test_domain_detection test_domain_filter test_domain_trainer test_domain_metrics; do
            TOTAL=\$((TOTAL + 1))
            if python3 testing/\${test}.py 2>&1 | tail -5 | grep -q '✅.*PASSED\|Summary.*Passed'; then
                echo \"✅ \${test}: PASSED\"
                PASSED=\$((PASSED + 1))
            else
                echo \"❌ \${test}: FAILED\"
                FAILED=\$((FAILED + 1))
            fi
        done
        echo ''
        
        # Week 2
        echo '=== Week 2: Integration Tests ==='
        for test in test_extraction_flow test_training_flow test_ab_testing_flow test_rollback_flow; do
            TOTAL=\$((TOTAL + 1))
            if python3 testing/\${test}.py 2>&1 | tail -5 | grep -q '✅.*PASSED\|Summary.*Passed'; then
                echo \"✅ \${test}: PASSED\"
                PASSED=\$((PASSED + 1))
            else
                echo \"❌ \${test}: FAILED\"
                FAILED=\$((FAILED + 1))
            fi
        done
        echo ''
        
        # Week 3
        echo '=== Week 3: Pattern Learning & Intelligence ==='
        for test in test_pattern_learning test_extraction_intelligence test_automation; do
            TOTAL=\$((TOTAL + 1))
            if python3 testing/\${test}.py 2>&1 | tail -5 | grep -q '✅.*PASSED\|Summary.*Passed'; then
                echo \"✅ \${test}: PASSED\"
                PASSED=\$((PASSED + 1))
            else
                echo \"❌ \${test}: FAILED\"
                FAILED=\$((FAILED + 1))
            fi
        done
        echo ''
        
        # Week 4
        echo '=== Week 4: Performance & Load Tests ==='
        for test in test_performance test_load test_concurrent_requests performance_benchmark; do
            TOTAL=\$((TOTAL + 1))
            if python3 testing/\${test}.py 2>&1 | tail -5 | grep -q '✅.*PASSED\|Summary.*Passed'; then
                echo \"✅ \${test}: PASSED\"
                PASSED=\$((PASSED + 1))
            else
                echo \"❌ \${test}: FAILED\"
                FAILED=\$((FAILED + 1))
            fi
        done
        echo ''
        
        # Integration
        echo '=== Integration Suite ==='
        TOTAL=\$((TOTAL + 1))
        if python3 testing/test_localai_integration_suite.py 2>&1 | tail -5 | grep -q '✅.*PASSED\|Summary.*Passed'; then
            echo \"✅ test_localai_integration_suite: PASSED\"
            PASSED=\$((PASSED + 1))
        else
            echo \"❌ test_localai_integration_suite: FAILED\"
            FAILED=\$((FAILED + 1))
        fi
        echo ''
        
        echo '============================================================'
        echo 'Test Summary'
        echo '============================================================'
        echo \"Total: \$TOTAL\"
        echo \"✅ Passed: \$PASSED\"
        echo \"❌ Failed: \$FAILED\"
        echo ''
        
        if [ \$FAILED -eq 0 ]; then
            echo '✅ ALL TESTS PASSED'
            exit 0
        else
            echo '❌ SOME TESTS FAILED'
            exit 1
        fi
    " 2>&1

