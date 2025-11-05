#!/bin/bash
# Run all tests with proper environment configuration

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running All Tests - Complete Suite"
echo "=========================================="
echo ""

# Set environment variables
export LOCALAI_URL="http://localhost:8081"
export EXTRACT_SERVICE_URL="http://localhost:19080"
export TRAINING_SERVICE_URL="http://localhost:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/amodels"
export REDIS_URL="redis://localhost:6379/0"
export NEO4J_URI="bolt://localhost:7687"

echo "Environment configured:"
echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  EXTRACT_SERVICE_URL: $EXTRACT_SERVICE_URL"
echo ""

# Week 1 Tests
echo "=== Week 1: Foundation Tests ==="
python3 testing/test_domain_detection.py 2>&1 | tail -10
python3 testing/test_domain_filter.py 2>&1 | tail -10
python3 testing/test_domain_trainer.py 2>&1 | tail -10
python3 testing/test_domain_metrics.py 2>&1 | tail -10

# Week 2 Tests
echo ""
echo "=== Week 2: Integration Tests ==="
python3 testing/test_extraction_flow.py 2>&1 | tail -10
python3 testing/test_training_flow.py 2>&1 | tail -10
python3 testing/test_ab_testing_flow.py 2>&1 | tail -10
python3 testing/test_rollback_flow.py 2>&1 | tail -10

# Week 3 Tests
echo ""
echo "=== Week 3: Phase 7-9 Tests ==="
python3 testing/test_pattern_learning.py 2>&1 | tail -10
python3 testing/test_extraction_intelligence.py 2>&1 | tail -10
python3 testing/test_automation.py 2>&1 | tail -10

# Week 4 Tests
echo ""
echo "=== Week 4: Performance Tests ==="
python3 testing/test_performance.py 2>&1 | tail -10
python3 testing/test_load.py 2>&1 | tail -10
python3 testing/test_concurrent_requests.py 2>&1 | tail -10

echo ""
echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="
