#!/bin/bash
# Run tests from within Docker network context

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running Tests from Docker Network"
echo "=========================================="
echo ""

# Run tests from within a container that has network access
docker compose -f infrastructure/docker/brev/docker-compose.yml run --rm \
  -e LOCALAI_URL=http://localai:8081 \
  -e DEEPAGENTS_URL=http://deepagents-service:9004 \
  -e GRAPH_URL=http://graph-server:8080 \
  -e SEARCH_URL=http://search-inference:8090 \
  -e EXTRACT_URL=http://extract-service:8082 \
  -e GATEWAY_URL=http://gateway:8000 \
  -e TRANSFORMERS_URL=http://transformers-service:9090 \
  -v "$(pwd)/testing:/workspace/testing:ro" \
  -v "$(pwd)/scripts:/workspace/scripts:ro" \
  trainer \
  sh -c "
    cd /workspace/testing &&
    pip install -q httpx &&
    python3 test_localai_integration_suite.py
  "

