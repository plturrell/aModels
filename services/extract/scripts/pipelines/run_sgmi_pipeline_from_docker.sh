#!/usr/bin/env bash
# Run SGMI pipeline from within Docker network
# This script runs the pipeline inside a container on the same network as services
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${ROOT_DIR}/../.." && pwd)

# Use container names (accessible from Docker network)
export EXTRACT_SERVICE_URL="http://extract-service:8082"
export DMS_SERVICE_URL="http://dms-service:8080"
export CATALOG_SERVICE_URL="http://catalog:8084"

echo "=== Running SGMI Pipeline from Docker Network ==="
echo ""
echo "Service URLs (Docker network):"
echo "  Extract: ${EXTRACT_SERVICE_URL}"
echo "  DMS: ${DMS_SERVICE_URL}"
echo "  Catalog: ${CATALOG_SERVICE_URL}"
echo ""

# Check if we're already in a Docker container
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container - executing pipeline directly..."
    "${SCRIPT_DIR}/run_sgmi_complete_pipeline.sh"
else
    echo "Running from host - executing via Docker container..."
    cd "${REPO_ROOT}"
    docker run --rm \
        --network brev_default \
        -v "${REPO_ROOT}:/workspace" \
        -w /workspace/services/extract \
        -e EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL}" \
        -e DMS_SERVICE_URL="${DMS_SERVICE_URL}" \
        -e CATALOG_SERVICE_URL="${CATALOG_SERVICE_URL}" \
        curlimages/curl:latest \
        sh -c "
            apk add --no-cache bash python3 py3-pip >/dev/null 2>&1 || true
            pip3 install requests >/dev/null 2>&1 || true
            bash scripts/pipelines/run_sgmi_complete_pipeline.sh
        "
fi

