#!/usr/bin/env bash
# SSH Tunnel Setup Script for All Services
# Sets up SSH tunnels for all major services

set -euo pipefail

# Configuration
SERVER_IP="${SERVER_IP:-54.196.0.75}"
SSH_USER="${SSH_USER:-$(whoami)}"

# Service port mappings: LOCAL_PORT:REMOTE_PORT
declare -A SERVICES=(
    ["neo4j_http"]="7474:7474"
    ["neo4j_bolt"]="7687:7687"
    ["graph"]="8080:8080"
    ["graph_admin"]="19080:19080"
    ["localai"]="8081:8081"
    ["extract"]="8082:8082"
    ["extract_grpc"]="9090:9090"
    ["extract_flight"]="8815:8815"
    ["search_inference"]="8090:8090"
    ["search_python"]="8091:8091"
    ["agentflow"]="8001:8001"
    ["browser"]="8070:8070"
    ["postgres"]="5432:5432"
    ["redis"]="6379:6379"
    ["elasticsearch"]="9200:9200"
)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}SSH Tunnel Setup for All Services${NC}"
echo "=========================================="
echo ""
echo "Server: ${SSH_USER}@${SERVER_IP}"
echo ""

# Build SSH command
SSH_ARGS=()
for service in "${!SERVICES[@]}"; do
    IFS=':' read -r local_port remote_port <<< "${SERVICES[$service]}"
    SSH_ARGS+=("-L" "${local_port}:localhost:${remote_port}")
    echo -e "${GREEN}✓${NC} ${service}: localhost:${local_port} -> ${SERVER_IP}:${remote_port}"
done

echo ""
echo "Services will be accessible at:"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - Graph Service: http://localhost:8080"
echo "  - LocalAI: http://localhost:8081"
echo "  - Extract: http://localhost:8082"
echo "  - Search Inference: http://localhost:8090"
echo "  - AgentFlow: http://localhost:8001"
echo "  - Browser: http://localhost:8070"
echo "  - Elasticsearch: http://localhost:9200"
echo ""
echo -e "${YELLOW}Starting tunnels (press Ctrl+C to stop)...${NC}"
echo ""

# Run SSH tunnel
ssh -N "${SSH_ARGS[@]}" "${SSH_USER}@${SERVER_IP}"

