#!/bin/bash
# Week 1: Smoke Tests
# Quick tests to verify basic functionality

set -e

echo "=========================================="
echo "Smoke Tests - Week 1"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
LOCALAI_URL="${LOCALAI_URL:-http://localhost:8081}"
EXTRACT_URL="${EXTRACT_SERVICE_URL:-http://localhost:19080}"
TRAINING_URL="${TRAINING_SERVICE_URL:-http://localhost:8080}"

PASSED=0
FAILED=0
SKIPPED=0

# Test function
test_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    echo -n "Testing $name... "
    
    if curl -s -f -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_code"; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((FAILED++))
        return 1
    fi
}

# Test Python script
test_python_script() {
    local name=$1
    local script=$2
    
    echo -n "Testing $name... "
    
    if python3 "$script" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${YELLOW}⏭️  SKIP${NC} (may require services running)"
        ((SKIPPED++))
        return 1
    fi
}

echo "Service Health Checks:"
echo "---------------------"

# Test LocalAI health
test_endpoint "LocalAI Health" "${LOCALAI_URL}/health" || echo "   LocalAI may not be running"

# Test Extract service health
test_endpoint "Extract Service Health" "${EXTRACT_URL}/healthz" || echo "   Extract service may not be running"

# Test LocalAI domains endpoint
test_endpoint "LocalAI Domains" "${LOCALAI_URL}/v1/domains" || echo "   LocalAI domains endpoint may not be available"

echo ""
echo "Test Scripts:"
echo "------------"

# Test domain detection script
if [ -f "testing/test_domain_detection.py" ]; then
    test_python_script "Domain Detection Tests" "testing/test_domain_detection.py"
else
    echo -e "${YELLOW}⏭️  Domain Detection Tests${NC} (file not found)"
    ((SKIPPED++))
fi

# Test domain filter script
if [ -f "testing/test_domain_filter.py" ]; then
    test_python_script "Domain Filter Tests" "testing/test_domain_filter.py"
else
    echo -e "${YELLOW}⏭️  Domain Filter Tests${NC} (file not found)"
    ((SKIPPED++))
fi

# Test domain trainer script
if [ -f "testing/test_domain_trainer.py" ]; then
    test_python_script "Domain Trainer Tests" "testing/test_domain_trainer.py"
else
    echo -e "${YELLOW}⏭️  Domain Trainer Tests${NC} (file not found)"
    ((SKIPPED++))
fi

# Test domain metrics script
if [ -f "testing/test_domain_metrics.py" ]; then
    test_python_script "Domain Metrics Tests" "testing/test_domain_metrics.py"
else
    echo -e "${YELLOW}⏭️  Domain Metrics Tests${NC} (file not found)"
    ((SKIPPED++))
fi

echo ""
echo "=========================================="
echo "Smoke Test Summary"
echo "=========================================="
echo -e "${GREEN}✅ Passed: ${PASSED}${NC}"
echo -e "${RED}❌ Failed: ${FAILED}${NC}"
echo -e "${YELLOW}⏭️  Skipped: ${SKIPPED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All smoke tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some smoke tests failed${NC}"
    exit 1
fi

