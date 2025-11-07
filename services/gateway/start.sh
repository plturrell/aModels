#!/bin/bash
# Start script for Gateway Service

set -e

echo "Starting aModels Gateway Service..."

# Set default port if not provided
export GATEWAY_PORT=${GATEWAY_PORT:-8000}

# Set default backend service URLs if not provided
export SEARCH_INFERENCE_URL=${SEARCH_INFERENCE_URL:-http://localhost:8090}
export GRAPH_SERVICE_URL=${GRAPH_SERVICE_URL:-http://localhost:8081}
export EXTRACT_URL=${EXTRACT_URL:-http://localhost:9002}
export CATALOG_URL=${CATALOG_URL:-http://localhost:8084}
export LOCALAI_URL=${LOCALAI_URL:-http://localhost:8080}
export AGENTFLOW_URL=${AGENTFLOW_URL:-http://localhost:9001}
export DEEP_RESEARCH_URL=${DEEP_RESEARCH_URL:-http://localhost:8085}
export PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY:-""}

echo "Configuration:"
echo "  GATEWAY_PORT: $GATEWAY_PORT"
echo "  SEARCH_INFERENCE_URL: $SEARCH_INFERENCE_URL"
echo "  GRAPH_SERVICE_URL: $GRAPH_SERVICE_URL"
echo "  EXTRACT_URL: $EXTRACT_URL"
echo "  CATALOG_URL: $CATALOG_URL"
echo "  LOCALAI_URL: $LOCALAI_URL"
echo "  AGENTFLOW_URL: $AGENTFLOW_URL"
echo "  DEEP_RESEARCH_URL: $DEEP_RESEARCH_URL"
echo "  PERPLEXITY_API_KEY: ${PERPLEXITY_API_KEY:0:10}..." # Show first 10 chars only

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting gateway on http://0.0.0.0:$GATEWAY_PORT"
echo "Health check: http://localhost:$GATEWAY_PORT/healthz"
echo ""

# Start uvicorn
exec uvicorn main:app --host 0.0.0.0 --port "$GATEWAY_PORT" --reload

