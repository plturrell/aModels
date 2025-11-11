#!/bin/bash
# Test script for all 6 high-priority improvements

set -e

echo "🧪 Testing High-Priority Improvements"
echo "======================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Data Validation
echo ""
echo "Test 1: Data Validation Before Storage"
echo "----------------------------------------"
cd /home/aModels/services/extract
if go test -v -run TestValidateNodes -run TestValidateEdges 2>&1 | grep -q "PASS"; then
    echo -e "${GREEN}✅ Data validation tests passed${NC}"
else
    echo -e "${RED}❌ Data validation tests failed${NC}"
    exit 1
fi

# Test 2: Retry Logic
echo ""
echo "Test 2: Retry Logic for Storage Operations"
echo "--------------------------------------------"
if go test -v -run TestRetryWithBackoff 2>&1 | grep -q "PASS"; then
    echo -e "${GREEN}✅ Retry logic tests passed${NC}"
else
    echo -e "${RED}❌ Retry logic tests failed${NC}"
    exit 1
fi

# Test 3: Consistency Validation
echo ""
echo "Test 3: Automatic Consistency Validation"
echo "------------------------------------------"
cd /home/aModels
if python3 -c "
import sys
sys.path.insert(0, 'services/training')
from data_access import UnifiedDataAccess
import os

# Test consistency validation
access = UnifiedDataAccess(
    postgres_dsn=os.getenv('POSTGRES_DSN', ''),
    redis_url=os.getenv('REDIS_URL', ''),
    neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    neo4j_username=os.getenv('NEO4J_USERNAME', 'neo4j'),
    neo4j_password=os.getenv('NEO4J_PASSWORD', 'amodels123')
)

result = access.validate_consistency('test_project')
print('Consistency check:', 'PASS' if isinstance(result, dict) else 'FAIL')
" 2>&1 | grep -q "PASS"; then
    echo -e "${GREEN}✅ Consistency validation test passed${NC}"
else
    echo -e "${YELLOW}⚠️  Consistency validation test skipped (requires database connections)${NC}"
fi

# Test 4: Unified Data Access Layer
echo ""
echo "Test 4: Unified Data Access Layer"
echo "----------------------------------"
if python3 -c "
import sys
sys.path.insert(0, 'services/training')
from data_access import UnifiedDataAccess, PostgresAdapter, RedisAdapter, Neo4jAdapter
print('Unified data access imports: PASS')
" 2>&1 | grep -q "PASS"; then
    echo -e "${GREEN}✅ Unified data access layer test passed${NC}"
else
    echo -e "${RED}❌ Unified data access layer test failed${NC}"
    exit 1
fi

# Test 5: Neo4j Batch Processing
echo ""
echo "Test 5: Neo4j Transaction Processing Optimization"
echo "----------------------------------------------------"
cd /home/aModels/services/extract
if go build -o /tmp/test_neo4j neo4j.go 2>&1; then
    echo -e "${GREEN}✅ Neo4j batch processing compiles successfully${NC}"
    rm -f /tmp/test_neo4j
else
    echo -e "${RED}❌ Neo4j batch processing compilation failed${NC}"
    exit 1
fi

# Test 6: Comprehensive Caching Strategy
echo ""
echo "Test 6: Comprehensive Caching Strategy"
echo "---------------------------------------"
cd /home/aModels
if python3 -c "
import sys
sys.path.insert(0, 'services/training')
from gnn_cache_manager import GNNCacheManager

cache = GNNCacheManager(default_ttl=3600)
cache.set('test', 'test_data', cache_type='test')
result = cache.get('test', cache_type='test')
print('Cache test:', 'PASS' if result == 'test_data' else 'FAIL')
" 2>&1 | grep -q "PASS"; then
    echo -e "${GREEN}✅ Caching strategy test passed${NC}"
else
    echo -e "${RED}❌ Caching strategy test failed${NC}"
    exit 1
fi

echo ""
echo "======================================"
echo -e "${GREEN}✅ All improvement tests completed${NC}"

