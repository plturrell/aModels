#!/bin/bash
# Run all tests with Step 0 service check first
# This ensures all services are ready before running tests

set -e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Running Tests with Service Health Check"
echo "============================================================"
echo ""

# Step 0: Check services
echo "Running Step 0: Service Health Check..."
if ! ./scripts/testing/00_check_services.sh; then
    echo ""
    echo "❌ Service health check failed!"
    echo "Please fix service issues before running tests."
    exit 1
fi

echo ""
echo "============================================================"
echo "All Services Ready - Running Tests"
echo "============================================================"
echo ""

# Export environment variables from service check
export LOCALAI_URL="${LOCALAI_URL:-http://localhost:8081}"
export EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL:-http://localhost:19080}"
export TRAINING_SERVICE_URL="${TRAINING_SERVICE_URL:-http://localhost:8080}"

# Run all tests
./scripts/testing/run_all_tests.sh

