#!/usr/bin/env bash
# Test all services accessibility via public IP

set -euo pipefail

SERVER_IP="${SERVER_IP:-54.196.0.75}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Testing Service Accessibility on ${SERVER_IP}"
echo "=============================================="
echo ""

# Test function
test_service() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Testing ${name}... "
    
    if response=$(curl -s -w "\n%{http_code}" --max-time 3 "${url}" 2>&1); then
        http_code=$(echo "$response" | tail -1)
        body=$(echo "$response" | sed '$d')
        
        if [ "$http_code" = "$expected_status" ] || [ "$http_code" = "404" ] && [[ "$body" == *"detail"* ]] || [[ "$body" == *"{"* ]]; then
            echo -e "${GREEN}✓ Accessible${NC} (HTTP ${http_code})"
            if [ -n "$body" ] && [ ${#body} -lt 100 ]; then
                echo "  Response: ${body}"
            fi
            return 0
        else
            echo -e "${YELLOW}⚠ Responding${NC} (HTTP ${http_code})"
            return 1
        fi
    else
        echo -e "${RED}✗ Not accessible${NC}"
        return 1
    fi
}

# Test services
test_service "Neo4j Browser" "http://${SERVER_IP}:7474"
test_service "Neo4j (root)" "http://${SERVER_IP}:7474/db"
test_service "LocalAI Models" "http://${SERVER_IP}:8081/v1/models"
test_service "Extract Health" "http://${SERVER_IP}:8082/healthz"
test_service "Search Inference" "http://${SERVER_IP}:8090"
test_service "AgentFlow Health" "http://${SERVER_IP}:8001/healthz"
test_service "Browser Automation Health" "http://${SERVER_IP}:8070/healthz"
test_service "Graph Service" "http://${SERVER_IP}:8080/healthz"
test_service "Elasticsearch" "http://${SERVER_IP}:9200"

echo ""
echo "=== Summary ==="
echo "✅ Neo4j: http://${SERVER_IP}:7474"
echo "   Bolt: bolt://ec2-54-196-0-75.compute-1.amazonaws.com:7687"
echo ""
echo "For detailed service URLs, see: docs/SERVICE_URLS.md"

