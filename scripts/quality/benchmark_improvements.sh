#!/bin/bash
# Performance benchmarking script for all 6 improvements

set -e

echo "📊 Benchmarking High-Priority Improvements"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL:-http://localhost:19080}"
RESULTS_DIR="${RESULTS_DIR:-./benchmark_results}"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.json"

echo "Extract Service URL: $EXTRACT_SERVICE_URL"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Function to make API call and measure time
benchmark_endpoint() {
    local endpoint=$1
    local name=$2
    local method=${3:-GET}
    local data=${4:-}
    
    echo -n "Benchmarking $name... "
    
    if [ "$method" = "GET" ]; then
        start_time=$(date +%s.%N)
        response=$(curl -s -w "\n%{http_code}" "$EXTRACT_SERVICE_URL$endpoint" || echo "ERROR")
        end_time=$(date +%s.%N)
    else
        start_time=$(date +%s.%N)
        response=$(curl -s -w "\n%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$data" "$EXTRACT_SERVICE_URL$endpoint" || echo "ERROR")
        end_time=$(date +%s.%N)
    fi
    
    duration=$(echo "$end_time - $start_time" | bc)
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        echo -e "${GREEN}✓${NC} (${duration}s, HTTP $http_code)"
        echo "{\"name\": \"$name\", \"endpoint\": \"$endpoint\", \"duration\": $duration, \"http_code\": $http_code, \"success\": true}" >> "$RESULTS_FILE"
    else
        echo -e "${YELLOW}⚠${NC} (${duration}s, HTTP $http_code)"
        echo "{\"name\": \"$name\", \"endpoint\": \"$endpoint\", \"duration\": $duration, \"http_code\": $http_code, \"success\": false}" >> "$RESULTS_FILE"
    fi
}

# Initialize results file
echo "[" > "$RESULTS_FILE"

# Benchmark 1: Metrics Endpoint
echo ""
echo "1. Testing Metrics Collection"
echo "-----------------------------"
benchmark_endpoint "/metrics/improvements" "Improvements Metrics"

# Benchmark 2: Health Check
echo ""
echo "2. Testing Health Check"
echo "----------------------"
benchmark_endpoint "/healthz" "Health Check"

# Close JSON array
sed -i '$ s/$/]/' "$RESULTS_FILE"

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Benchmarking completed${NC}"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To view metrics:"
echo "  curl $EXTRACT_SERVICE_URL/metrics/improvements | jq"

