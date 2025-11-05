#!/bin/bash
# Run Step 0 first, then all tests
# This ensures all services are ready before running tests

set +e  # Don't exit on error

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Step 0 + All Tests"
echo "============================================================"
echo ""

# Step 0: Check services
echo "Running Step 0: Service Health Check..."
if ./testing/00_check_services.sh; then
    echo ""
    echo "✅ All services are ready!"
    echo ""
    
    # Export environment variables from Step 0
    export LOCALAI_URL="${LOCALAI_URL:-http://localhost:8081}"
    export EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL:-http://localhost:19080}"
    export TRAINING_SERVICE_URL="${TRAINING_SERVICE_URL:-http://localhost:8080}"
    
    # If LocalAI is only accessible from Docker network, update URL
    if ! python3 -c "import httpx; httpx.get('$LOCALAI_URL/readyz', timeout=2)" 2>/dev/null; then
        echo "⚠️  LocalAI not accessible from host, using Docker network URL for tests"
        export LOCALAI_URL="http://localai:8080"
        echo "   Using: $LOCALAI_URL"
    fi
    
    echo ""
    echo "Running all tests..."
    echo ""
    
    # Run all tests
    ./testing/run_all_tests_working.sh
else
    echo ""
    echo "❌ Step 0 failed - services are not ready"
    echo "Please fix service issues before running tests"
    exit 1
fi

