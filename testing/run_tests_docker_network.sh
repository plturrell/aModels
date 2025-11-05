#!/bin/bash
# Run tests using Docker network URLs (for services accessible from Docker network)

set +e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Running Tests with Docker Network URLs"
echo "============================================================"
echo ""

# Use Docker network URLs since services are accessible from Docker network
export LOCALAI_URL="http://localai:8080"
export EXTRACT_SERVICE_URL="http://extract-service:19080"
export TRAINING_SERVICE_URL="http://training-service:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
export REDIS_URL="redis://redis:6379/0"
export NEO4J_URI="bolt://neo4j:7687"

echo "Using Docker network URLs:"
echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  EXTRACT_SERVICE_URL: $EXTRACT_SERVICE_URL"
echo ""

# Run tests from training-shell container where services are accessible
echo "Running tests from Docker container..."
docker exec training-shell bash -c "
cd /workspace || cd /home/aModels
export LOCALAI_URL='http://localai:8080'
export EXTRACT_SERVICE_URL='http://extract-service:19080'

# Week 1 Tests
echo '=== Week 1: Foundation Tests ==='
python3 testing/test_domain_detection.py 2>&1 | tail -10
python3 testing/test_domain_filter.py 2>&1 | tail -10
python3 testing/test_domain_trainer.py 2>&1 | tail -10
python3 testing/test_domain_metrics.py 2>&1 | tail -10
" 2>&1 || echo "⚠️  Tests need to be run from container or host with proper URLs"

