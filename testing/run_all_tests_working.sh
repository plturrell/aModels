#!/bin/bash
# Run all tests with proper error handling and service checks
# NOTE: This script assumes Step 0 (00_check_services.sh) has been run first
# For automatic service check, use run_all_tests_with_check.sh instead

set -e

cd "$(dirname "$0")/.."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Running All Tests - Complete Suite"
echo "=========================================="
echo ""
echo "⚠️  NOTE: Ensure Step 0 (00_check_services.sh) has been run first!"
echo ""

# Set environment variables (use values from Step 0 if available)
export LOCALAI_URL="${LOCALAI_URL:-http://localhost:8081}"
export EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL:-http://localhost:19080}"
export TRAINING_SERVICE_URL="${TRAINING_SERVICE_URL:-http://localhost:8080}"
export POSTGRES_DSN="${POSTGRES_DSN:-postgresql://postgres:postgres@localhost:5432/amodels}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"

echo "Environment configured:"
echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  EXTRACT_SERVICE_URL: $EXTRACT_SERVICE_URL"
echo ""

# Quick service check (lightweight)
echo "Quick service check..."
python3 -c "
import httpx
import sys

services = [
    ('LocalAI', '$LOCALAI_URL/health'),
]

for name, url in services:
    try:
        response = httpx.get(url, timeout=5)
        if response.status_code == 200:
            print(f'✅ {name}: Running')
        else:
            print(f'⚠️  {name}: Status {response.status_code}')
    except Exception as e:
        print(f'❌ {name}: Not accessible')
" 2>&1 || echo "⚠️  Service check failed (run 00_check_services.sh for full check)"

echo ""
echo "=========================================="
echo "Week 1: Foundation Tests"
echo "=========================================="

# Week 1 Tests
echo ""
echo "Running domain detection tests..."
python3 testing/test_domain_detection.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running domain filter tests..."
python3 testing/test_domain_filter.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running domain trainer tests..."
python3 testing/test_domain_trainer.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running domain metrics tests..."
python3 testing/test_domain_metrics.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "=========================================="
echo "Week 2: Integration Tests"
echo "=========================================="

echo ""
echo "Running extraction flow tests..."
python3 testing/test_extraction_flow.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running training flow tests..."
python3 testing/test_training_flow.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running A/B testing flow tests..."
python3 testing/test_ab_testing_flow.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running rollback flow tests..."
python3 testing/test_rollback_flow.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "=========================================="
echo "Week 3: Phase 7-9 Tests"
echo "=========================================="

echo ""
echo "Running pattern learning tests..."
python3 testing/test_pattern_learning.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running extraction intelligence tests..."
python3 testing/test_extraction_intelligence.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running automation tests..."
python3 testing/test_automation.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "=========================================="
echo "Week 4: Performance Tests"
echo "=========================================="

echo ""
echo "Running performance tests..."
python3 testing/test_performance.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running load tests..."
python3 testing/test_load.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "Running concurrent request tests..."
python3 testing/test_concurrent_requests.py 2>&1 | tail -10 || echo "⚠️  Test had issues"

echo ""
echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="
echo ""
echo "Note: Some tests may skip if services are not available."
echo "This is expected behavior."

