#!/bin/bash
# Run all tests from Docker container where services are accessible
# This script runs tests inside the training-shell container

set -e

echo "=========================================="
echo "Running All Tests from Docker Container"
echo "=========================================="
echo ""

# Set environment variables for Docker service names
export LOCALAI_URL="http://localai:8080"
export EXTRACT_SERVICE_URL="http://extract-service:19080"
export TRAINING_SERVICE_URL="http://training-service:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
export REDIS_URL="redis://redis:6379/0"
export NEO4J_URI="bolt://neo4j:7687"

echo "Environment configured for Docker network:"
echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  EXTRACT_SERVICE_URL: $EXTRACT_SERVICE_URL"
echo ""

# Week 1: Foundation Tests
echo "=== Week 1: Foundation Tests ==="
docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && export EXTRACT_SERVICE_URL='$EXTRACT_SERVICE_URL' && python3 testing/test_domain_detection.py" 2>&1 | tail -15
echo ""

docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && python3 testing/test_domain_filter.py" 2>&1 | tail -15
echo ""

docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && python3 testing/test_domain_trainer.py" 2>&1 | tail -15
echo ""

docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && python3 testing/test_domain_metrics.py" 2>&1 | tail -15
echo ""

# Week 2: Integration Tests
echo "=== Week 2: Integration Tests ==="
docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && export EXTRACT_SERVICE_URL='$EXTRACT_SERVICE_URL' && python3 testing/test_extraction_flow.py" 2>&1 | tail -15
echo ""

docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && python3 testing/test_training_flow.py" 2>&1 | tail -15
echo ""

docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && python3 testing/test_ab_testing_flow.py" 2>&1 | tail -15
echo ""

docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && python3 testing/test_rollback_flow.py" 2>&1 | tail -15
echo ""

# Week 3: Phase 7-9 Tests
echo "=== Week 3: Phase 7-9 Tests ==="
docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && python3 testing/test_pattern_learning.py" 2>&1 | tail -15
echo ""

docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && export EXTRACT_SERVICE_URL='$EXTRACT_SERVICE_URL' && python3 testing/test_extraction_intelligence.py" 2>&1 | tail -15
echo ""

docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && python3 testing/test_automation.py" 2>&1 | tail -15
echo ""

# Week 4: Performance Tests
echo "=== Week 4: Performance Tests ==="
docker exec training-shell bash -c "cd /workspace && export LOCALAI_URL='$LOCALAI_URL' && export EXTRACT_SERVICE_URL='$EXTRACT_SERVICE_URL' && python3 testing/test_performance.py" 2>&1 | tail -15
echo ""

echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="
