#!/bin/bash
# Validate Test Migration
# Checks that all expected tests exist in new locations

set -e
cd "$(dirname "$0")/.."

echo "Validating test migration..."
echo ""

MISSING=0

check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Missing: $1"
        ((MISSING++))
    else
        echo "✅ Found: $1"
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "❌ Missing directory: $1"
        ((MISSING++))
    else
        echo "✅ Found directory: $1"
    fi
}

echo "Checking directories..."
check_dir "integration/services"
check_dir "integration/workflows"
check_dir "e2e"
check_dir "performance"
check_dir "benchmarks"
check_dir "domain"
check_dir "fixtures"
check_dir "scripts"

echo ""
echo "Checking key integration tests..."
check_file "integration/services/test_localai_integration.py"
check_file "integration/services/test_gnn_agent_integration.py"
check_file "integration/workflows/test_extraction_flow.py"
check_file "integration/workflows/test_training_flow.py"

echo ""
echo "Checking performance tests..."
check_file "performance/test_performance.py"
check_file "performance/test_load.py"

echo ""
echo "Checking scripts..."
check_file "scripts/run_all_tests.sh"
check_file "scripts/run_integration_tests.sh"
check_file "scripts/check_services.sh"

echo ""
if [ $MISSING -eq 0 ]; then
    echo "✅ All files present!"
    exit 0
else
    echo "❌ $MISSING files/directories missing"
    exit 1
fi
