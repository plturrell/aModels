#!/bin/bash
set -e

echo "=== DMS Service Comprehensive Test Suite ==="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
pass_test() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((TESTS_PASSED++))
}

fail_test() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((TESTS_FAILED++))
}

info() {
    echo -e "${YELLOW}ℹ INFO${NC}: $1"
}

# Test 1: Check if .env.example exists
echo "Test 1: Check .env.example exists"
if [ -f ".env.example" ]; then
    pass_test ".env.example file exists"
else
    fail_test ".env.example file not found"
fi

# Test 2: Check Alembic setup
echo ""
echo "Test 2: Check Alembic configuration"
if [ -f "alembic.ini" ] && [ -d "alembic/versions" ]; then
    pass_test "Alembic configuration complete"
else
    fail_test "Alembic configuration incomplete"
fi

# Test 3: Check Python imports
echo ""
echo "Test 3: Validate Python module imports"
if python3 -c "from app.core.auth import verify_token, create_access_token; from app.api.routers.health import health_check" 2>/dev/null; then
    pass_test "All Python imports valid"
else
    fail_test "Python import errors detected"
fi

# Test 4: Check required files
echo ""
echo "Test 4: Check documentation files"
REQUIRED_FILES=("README.md" "SECURITY.md" "CHANGELOG.md" "QUICKSTART.md")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        pass_test "$file exists"
    else
        fail_test "$file not found"
    fi
done

# Test 5: Check auth module
echo ""
echo "Test 5: Test auth module functions"
if python3 << 'PYEOF'
from app.core.auth import create_access_token, verify_jwt_token
token = create_access_token({"sub": "test"})
payload = verify_jwt_token(token)
assert payload is not None
assert payload["sub"] == "test"
print("Auth module works correctly")
PYEOF
then
    pass_test "Authentication module functional"
else
    fail_test "Authentication module has issues"
fi

# Test 6: Check health router
echo ""
echo "Test 6: Validate health check router"
if python3 -c "from app.api.routers.health import router; assert len(router.routes) >= 4" 2>/dev/null; then
    pass_test "Health check router has all endpoints"
else
    fail_test "Health check router incomplete"
fi

# Test 7: Validate migration file
echo ""
echo "Test 7: Check migration file syntax"
if python3 -c "import sys; sys.path.insert(0, 'alembic/versions'); import importlib; spec = importlib.util.spec_from_file_location('migration', 'alembic/versions/001_initial_schema.py'); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)" 2>/dev/null; then
    pass_test "Migration file is valid Python"
else
    fail_test "Migration file has syntax errors"
fi

# Summary
echo ""
echo "==================================="
echo "Test Summary"
echo "==================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
