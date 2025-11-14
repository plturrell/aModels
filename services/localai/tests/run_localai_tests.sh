#!/bin/bash
# LocalAI Integration Test Execution Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "LocalAI Integration Test Runner"
echo "=========================================="

# Check if services are running
echo ""
echo "Checking service availability..."

# Check LocalAI
if curl -f -s "${LOCALAI_SERVICE_URL:-http://localhost:8081}/healthz" > /dev/null 2>&1; then
    echo "✅ LocalAI service is running"
else
    echo "❌ LocalAI service is not accessible"
    echo "   Please ensure LocalAI is running on port 8081"
    exit 1
fi

# Check Transformers Service
if curl -f -s "${TRANSFORMERS_SERVICE_URL:-http://localhost:9090}/health" > /dev/null 2>&1; then
    echo "✅ Transformers service is running"
else
    echo "⚠️  Transformers service is not accessible (may be optional)"
fi

# Check Model Server (optional)
if curl -f -s "${MODEL_SERVER_URL:-http://model-server:8088}/health" > /dev/null 2>&1; then
    echo "✅ Model server is running"
else
    echo "⚠️  Model server is not accessible (may be optional)"
fi

echo ""
echo "Running LocalAI integration tests..."
echo ""

# Set environment variables
export LOCALAI_SERVICE_URL="${LOCALAI_SERVICE_URL:-http://localhost:8081}"
export TRANSFORMERS_SERVICE_URL="${TRANSFORMERS_SERVICE_URL:-http://localhost:9090}"
export MODEL_SERVER_URL="${MODEL_SERVER_URL:-http://model-server:8088}"
export TEST_TIMEOUT="${TEST_TIMEOUT:-120}"
export RESULTS_DIR="${RESULTS_DIR:-/tmp/localai_test_results}"

# Run the test script
python3 test_localai_integration.py

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="

