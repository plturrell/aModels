#!/bin/bash
# Run Performance Tests

set -e
cd "$(dirname "$0")/../performance"

echo "=========================================="
echo "Running Performance Tests"
echo "=========================================="
echo ""

for test in test_*.py performance_benchmark.py; do
    if [ -f "$test" ]; then
        echo "Running $test..."
        python3 "$test" || echo "⚠️  $test failed"
    fi
done

echo ""
echo "✅ Performance tests complete"
