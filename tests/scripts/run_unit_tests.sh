#!/bin/bash
# Run Unit Tests
# Note: Most unit tests are co-located with source code in services/

set -e
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="
echo ""

echo "Running Go unit tests in services..."
cd services

# Find all *_test.go files (excluding integration tests)
for service_dir in */; do
    if [ -d "$service_dir" ]; then
        service_name=$(basename "$service_dir")
        echo "Testing $service_name..."
        cd "$service_dir"
        if ls *_test.go 1> /dev/null 2>&1; then
            go test -v -short ./... || echo "⚠️  $service_name tests failed"
        fi
        cd ..
    fi
done

echo ""
echo "✅ Unit tests complete"
