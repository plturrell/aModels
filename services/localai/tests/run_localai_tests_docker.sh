#!/bin/bash
# LocalAI Integration Test Execution Script (Docker)
# Runs tests from within the LocalAI container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="/tmp/localai_tests"
RESULTS_DIR="/tmp/localai_test_results"

echo "=========================================="
echo "LocalAI Integration Test Runner (Docker)"
echo "=========================================="

# Create test directory
mkdir -p "$TEST_DIR"
mkdir -p "$RESULTS_DIR"

# Copy test files to test directory
if [ -f "$SCRIPT_DIR/test_localai_integration.py" ]; then
    cp "$SCRIPT_DIR/test_localai_integration.py" "$TEST_DIR/"
    echo "✅ Copied test_localai_integration.py"
else
    echo "⚠️  test_localai_integration.py not found at $SCRIPT_DIR"
fi

if [ -f "$SCRIPT_DIR/test_queries.json" ]; then
    cp "$SCRIPT_DIR/test_queries.json" "$TEST_DIR/"
    echo "✅ Copied test_queries.json"
else
    echo "⚠️  test_queries.json not found at $SCRIPT_DIR"
fi

# Check if domains.json is accessible
if [ -f "/config/domains.json" ]; then
    echo "✅ Found domains.json at /config/domains.json"
elif [ -f "/workspace/services/localai/config/domains.json" ]; then
    echo "✅ Found domains.json at /workspace/services/localai/config/domains.json"
else
    echo "⚠️  domains.json not found in expected locations"
fi

echo ""
echo "Checking service availability..."
echo ""

# Check LocalAI (same container)
if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" 2>/dev/null; then
    echo "✅ LocalAI service is running"
else
    echo "❌ LocalAI service is not accessible"
    exit 1
fi

# Check Transformers Service
if python3 -c "import urllib.request; urllib.request.urlopen('http://transformers-service:9090/health')" 2>/dev/null; then
    echo "✅ Transformers service is running"
else
    echo "⚠️  Transformers service is not accessible (may be optional)"
fi

# Check Model Server (optional)
if python3 -c "import urllib.request; urllib.request.urlopen('http://model-server:8088/health')" 2>/dev/null; then
    echo "✅ Model server is running"
else
    echo "⚠️  Model server is not accessible (may be optional)"
fi

echo ""
echo "Running LocalAI integration tests..."
echo ""

# Set environment variables
export LOCALAI_SERVICE_URL="http://localhost:8080"
export TRANSFORMERS_SERVICE_URL="http://transformers-service:9090"
export MODEL_SERVER_URL="http://model-server:8088"
export TEST_TIMEOUT="${TEST_TIMEOUT:-120}"
export RESULTS_DIR="$RESULTS_DIR"

# Change to test directory
cd "$TEST_DIR"

# Run the test script
python3 test_localai_integration.py

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"

