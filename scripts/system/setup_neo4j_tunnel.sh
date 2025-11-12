#!/usr/bin/env bash
# SSH Tunnel Setup Script for Neo4j Browser
# This script sets up an SSH tunnel to access Neo4j from your local machine

set -euo pipefail

# Configuration
SERVER_IP="${NEO4J_SERVER_IP:-54.196.0.75}"
SSH_USER="${NEO4J_SSH_USER:-$(whoami)}"
LOCAL_PORT_HTTP=7474
LOCAL_PORT_BOLT=7687
REMOTE_PORT_HTTP=7474
REMOTE_PORT_BOLT=7687

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Neo4j SSH Tunnel Setup${NC}"
echo "================================"
echo ""
echo "Server: ${SSH_USER}@${SERVER_IP}"
echo "Local ports:"
echo "  - HTTP (Browser): ${LOCAL_PORT_HTTP} -> ${SERVER_IP}:${REMOTE_PORT_HTTP}"
echo "  - Bolt (Database): ${LOCAL_PORT_BOLT} -> ${SERVER_IP}:${REMOTE_PORT_BOLT}"
echo ""

# Check if ports are already in use
check_port() {
    local port=$1
    if lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port ${port} is already in use${NC}"
        read -p "Do you want to kill the process using port ${port}? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            lsof -ti:${port} | xargs kill -9 2>/dev/null || true
            sleep 1
        else
            echo -e "${RED}Port ${port} is in use. Please free it or choose different ports.${NC}"
            exit 1
        fi
    fi
}

check_port ${LOCAL_PORT_HTTP}
check_port ${LOCAL_PORT_BOLT}

# Test SSH connection
echo -e "${GREEN}Testing SSH connection...${NC}"
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes ${SSH_USER}@${SERVER_IP} echo "Connection OK" >/dev/null 2>&1; then
    echo -e "${YELLOW}SSH connection test failed. You may need to enter your password.${NC}"
fi

# Create SSH tunnel in background
echo -e "${GREEN}Creating SSH tunnel...${NC}"
echo ""
echo "To stop the tunnel, press Ctrl+C or run:"
echo "  pkill -f 'ssh.*${LOCAL_PORT_HTTP}:localhost:${REMOTE_PORT_HTTP}'"
echo ""
echo "Once the tunnel is active, access Neo4j Browser at:"
echo -e "${GREEN}  http://localhost:${LOCAL_PORT_HTTP}${NC}"
echo ""
echo "Credentials:"
echo "  Username: neo4j"
echo "  Password: amodels123"
echo ""
echo -e "${YELLOW}Starting tunnel (press Ctrl+C to stop)...${NC}"
echo ""

# Run SSH tunnel
ssh -N -L ${LOCAL_PORT_HTTP}:localhost:${REMOTE_PORT_HTTP} \
        -L ${LOCAL_PORT_BOLT}:localhost:${REMOTE_PORT_BOLT} \
        ${SSH_USER}@${SERVER_IP}

