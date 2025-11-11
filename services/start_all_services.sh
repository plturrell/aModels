#!/bin/bash
# Start all aModels services
# This script starts: Gateway, Search Inference, and LocalAI

set -e

echo "=========================================="
echo "Starting aModels Services"
echo "=========================================="
echo ""

# Get the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -tlnp 2>/dev/null | grep -q ":$port " || ss -tlnp 2>/dev/null | grep -q ":$port "; then
        return 0
    else
        return 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=90
    local attempt=0
    
    echo -n "Waiting for $name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 1
    done
    echo -e " ${RED}✗${NC} (timeout)"
    return 1
}

# 1. Start Gateway Service
echo "1. Starting Gateway Service..."
if check_port 8000; then
    echo -e "   ${YELLOW}Gateway already running on port 8000${NC}"
else
    cd services/gateway
    nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > gateway.log 2>&1 &
    GATEWAY_PID=$!
    echo "   Gateway started (PID: $GATEWAY_PID)"
    cd "$BASE_DIR"
    wait_for_service "http://localhost:8000/healthz" "Gateway"
fi

# 2. Start Search Inference Service
echo ""
echo "2. Starting Search Inference Service..."
if check_port 8090; then
    echo -e "   ${YELLOW}Search Inference already running on port 8090${NC}"
else
    cd services/search/search-inference
    # Ensure module graph is up to date for Go 1.18 toolchain
    env GOWORK=off go mod tidy >> ../../search-inference.log 2>&1 || true
    nohup env GOWORK=off go run ./cmd/search-server/main.go -port 8090 > ../../search-inference.log 2>&1 &
    SEARCH_PID=$!
    echo "   Search Inference started (PID: $SEARCH_PID)"
    cd "$BASE_DIR"
    wait_for_service "http://localhost:8090/health" "Search Inference" || echo -e " ${YELLOW}(optional) continuing without Search Inference${NC}"
fi

# 3. Start LocalAI Service
echo ""
echo "3. Starting LocalAI Service..."
if check_port 8080; then
    echo -e "   ${YELLOW}LocalAI already running on port 8080${NC}"
else
    cd services/localai
    if [ -f scripts/start_localai_stack.sh ]; then
        chmod +x scripts/start_localai_stack.sh
        ./scripts/start_localai_stack.sh > localai.log 2>&1 &
        LOCALAI_PID=$!
        echo "   LocalAI stack started (PID: $LOCALAI_PID)"
        cd "$BASE_DIR"
        sleep 5  # Give LocalAI more time to start
        wait_for_service "http://localhost:8080/v1/models" "LocalAI"
    else
        echo -e "   ${RED}LocalAI startup script not found${NC}"
    fi
fi

# 4. Verify all services
echo ""
echo "=========================================="
echo "Service Status"
echo "=========================================="

check_service() {
    local name=$1
    local url=$2
    if curl -s "$url" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name: Running"
        return 0
    else
        echo -e "${RED}✗${NC} $name: Not responding"
        return 1
    fi
}

check_service "Gateway" "http://localhost:8000/healthz"
check_service "Search Inference" "http://localhost:8090/health" || true
check_service "LocalAI" "http://localhost:8080/v1/models"

echo ""
echo "=========================================="
echo "Service URLs"
echo "=========================================="
echo "Gateway:        http://localhost:8000"
echo "Search:         http://localhost:8090"
echo "LocalAI:        http://localhost:8080"
echo ""
echo "Health Checks:"
echo "  Gateway:      http://localhost:8000/healthz"
echo "  Search:       http://localhost:8090/health"
echo "  LocalAI:      http://localhost:8080/v1/models"
echo ""

