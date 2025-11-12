#!/bin/bash
# Run End-to-End Tests

set -e
cd "$(dirname "$0")/../e2e"

echo "=========================================="
echo "Running End-to-End Tests"
echo "=========================================="
echo ""

# Check all services are running
if ! ../scripts/check_services.sh; then
    echo "❌ E2E tests require all services running"
    exit 1
fi

for test in test_*.sh; do
    if [ -f "$test" ]; then
        echo "Running $test..."
        bash "$test" || echo "⚠️  $test failed"
    fi
done

echo ""
echo "✅ E2E tests complete"
