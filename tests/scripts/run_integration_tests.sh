#!/bin/bash
# Run Integration Tests

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running Integration Tests"
echo "=========================================="
echo ""

# Check services
if ./scripts/check_services.sh; then
    echo "✅ Services ready"
else
    echo "⚠️  Some services not available"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Running service integration tests..."
cd integration/services
for test in test_*.py; do
    if [ -f "$test" ]; then
        echo "Running $test..."
        python3 "$test" || echo "⚠️  $test failed"
    fi
done

echo ""
echo "Running workflow integration tests..."
cd ../workflows
for test in test_*.py; do
    if [ -f "$test" ]; then
        echo "Running $test..."
        python3 "$test" || echo "⚠️  $test failed"
    fi
done

echo ""
echo "Running Go integration tests..."
cd ../services
go test -v . || echo "⚠️  Go tests failed"

echo ""
echo "✅ Integration tests complete"
