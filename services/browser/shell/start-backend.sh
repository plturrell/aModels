#!/bin/bash
# Start script for Backend Services (Gateway + Shell Server)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Starting aModels Backend Services..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if gateway is already running
if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Gateway appears to be already running on port 8000${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if shell server is already running
if curl -s http://localhost:4173 > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Shell server appears to be already running on port 4173${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set environment variables
export GATEWAY_PORT=${GATEWAY_PORT:-8000}
export SHELL_GATEWAY_URL=${SHELL_GATEWAY_URL:-http://localhost:8000}
export VITE_SHELL_API=${VITE_SHELL_API:-""}

# Backend service URLs (defaults)
export SEARCH_INFERENCE_URL=${SEARCH_INFERENCE_URL:-http://localhost:8090}
export GRAPH_SERVICE_URL=${GRAPH_SERVICE_URL:-http://localhost:8081}
export EXTRACT_URL=${EXTRACT_URL:-http://localhost:9002}
export CATALOG_URL=${CATALOG_URL:-http://localhost:8084}
export LOCALAI_URL=${LOCALAI_URL:-http://localhost:8080}
export AGENTFLOW_URL=${AGENTFLOW_URL:-http://localhost:9001}
export DEEP_RESEARCH_URL=${DEEP_RESEARCH_URL:-http://localhost:8085}

echo -e "${GREEN}Configuration:${NC}"
echo "  Gateway: http://localhost:$GATEWAY_PORT"
echo "  Shell Server: http://localhost:4173"
echo "  SHELL_GATEWAY_URL: $SHELL_GATEWAY_URL"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$GATEWAY_PID" ]; then
        kill $GATEWAY_PID 2>/dev/null || true
    fi
    if [ ! -z "$SHELL_PID" ]; then
        kill $SHELL_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Services stopped${NC}"
}

trap cleanup EXIT INT TERM

# Start Gateway Service
echo -e "${GREEN}Starting Gateway Service...${NC}"
cd "$REPO_ROOT/services/gateway"
if [ ! -f "start.sh" ]; then
    echo "Gateway start.sh not found, using uvicorn directly"
    uvicorn main:app --host 0.0.0.0 --port "$GATEWAY_PORT" --reload > /tmp/gateway.log 2>&1 &
else
    bash start.sh > /tmp/gateway.log 2>&1 &
fi
GATEWAY_PID=$!

# Wait for gateway to start
echo "Waiting for gateway to start..."
for i in {1..30}; do
    if curl -s http://localhost:$GATEWAY_PORT/healthz > /dev/null 2>&1; then
        echo -e "${GREEN}Gateway is running (PID: $GATEWAY_PID)${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Gateway failed to start${NC}"
        echo "Check logs: /tmp/gateway.log"
        exit 1
    fi
    sleep 1
done

# Build shell server if needed
echo -e "${GREEN}Building Shell Server...${NC}"
cd "$SCRIPT_DIR/cmd/server"
if [ ! -f "server" ] || [ "server" -ot "main.go" ]; then
    echo "Building shell server..."
    go build -o server main.go
fi
cd "$SCRIPT_DIR"

# Start Shell Server
echo -e "${GREEN}Starting Shell Server...${NC}"
./cmd/server/server -addr :4173 > /tmp/shell-server.log 2>&1 &
SHELL_PID=$!

# Wait for shell server to start
echo "Waiting for shell server to start..."
for i in {1..10}; do
    if curl -s http://localhost:4173 > /dev/null 2>&1; then
        echo -e "${GREEN}Shell Server is running (PID: $SHELL_PID)${NC}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "${RED}Shell server failed to start${NC}"
        echo "Check logs: /tmp/shell-server.log"
        exit 1
    fi
    sleep 1
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Backend Services Running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Gateway Service:"
echo "  URL: http://localhost:$GATEWAY_PORT"
echo "  Health: http://localhost:$GATEWAY_PORT/healthz"
echo "  Docs: http://localhost:$GATEWAY_PORT/docs"
echo ""
echo "Shell Server:"
echo "  URL: http://localhost:4173"
echo "  Frontend: http://localhost:4173 (serves UI)"
echo ""
echo "Logs:"
echo "  Gateway: /tmp/gateway.log"
echo "  Shell Server: /tmp/shell-server.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for interrupt
wait

