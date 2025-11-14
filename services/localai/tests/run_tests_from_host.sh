#!/bin/bash
# Host-side wrapper script to run LocalAI integration tests
# Copies test files to LocalAI container and executes tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${LOCALAI_CONTAINER:-localai}"

echo "=========================================="
echo "LocalAI Integration Test Runner (Host)"
echo "=========================================="

# Check if container is running
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ Container '$CONTAINER_NAME' is not running"
    echo "   Please start the LocalAI container first"
    exit 1
fi

echo "✅ Container '$CONTAINER_NAME' is running"
echo ""

# Check if container has Python
if ! docker exec "$CONTAINER_NAME" which python3 > /dev/null 2>&1; then
    echo "❌ Container '$CONTAINER_NAME' does not have Python3 installed"
    echo "   Please rebuild the container with Python support"
    exit 1
fi

echo "✅ Container has Python3"
echo ""

# Copy test files to container
echo "Copying test files to container..."
docker cp "$SCRIPT_DIR/test_localai_integration.py" "$CONTAINER_NAME:/tmp/localai_tests/" 2>/dev/null || \
    docker exec "$CONTAINER_NAME" mkdir -p /tmp/localai_tests && \
    docker cp "$SCRIPT_DIR/test_localai_integration.py" "$CONTAINER_NAME:/tmp/localai_tests/"

docker cp "$SCRIPT_DIR/test_queries.json" "$CONTAINER_NAME:/tmp/localai_tests/" 2>/dev/null || echo "⚠️  test_queries.json not found"

# Copy test execution script
docker cp "$SCRIPT_DIR/run_localai_tests_docker.sh" "$CONTAINER_NAME:/tmp/localai_tests/" 2>/dev/null || echo "⚠️  run_localai_tests_docker.sh not found"

echo "✅ Test files copied"
echo ""

# Execute tests in container
echo "Executing tests in container..."
echo ""

docker exec -e LOCALAI_SERVICE_URL="http://localhost:8080" \
           -e TRANSFORMERS_SERVICE_URL="http://transformers-service:9090" \
           -e MODEL_SERVER_URL="http://model-server:8088" \
           -e TEST_TIMEOUT="${TEST_TIMEOUT:-120}" \
           "$CONTAINER_NAME" \
           bash -c "cd /tmp/localai_tests && chmod +x run_localai_tests_docker.sh 2>/dev/null || true && bash run_localai_tests_docker.sh || python3 test_localai_integration.py"

# Copy results back to host
echo ""
echo "Copying test results from container..."
RESULTS_DIR="/tmp/localai_test_results"
HOST_RESULTS_DIR="$SCRIPT_DIR/../test_results"

mkdir -p "$HOST_RESULTS_DIR"
docker cp "$CONTAINER_NAME:$RESULTS_DIR/." "$HOST_RESULTS_DIR/" 2>/dev/null && \
    echo "✅ Results copied to $HOST_RESULTS_DIR" || \
    echo "⚠️  Could not copy results (they remain in container at $RESULTS_DIR)"

echo ""
echo "=========================================="
echo "Test execution complete!"
echo "=========================================="

