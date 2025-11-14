#!/usr/bin/env bash
# Run SGMI pipeline from within Docker network
# This script runs the pipeline inside a container on the same network as services
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

# Use container names (accessible from Docker network)
export EXTRACT_SERVICE_URL="http://extract-service:8082"
export CATALOG_SERVICE_URL="http://catalog:8084"

echo "=== Running SGMI Pipeline from Docker Network ==="
echo ""
echo "Service URLs (Docker network):"
echo "  Extract: ${EXTRACT_SERVICE_URL}"
echo "  Catalog: ${CATALOG_SERVICE_URL}"
echo ""

# Get absolute paths
REPO_ABS=$(cd "${REPO_ROOT}" && pwd)
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_sgmi_complete_pipeline.sh"
ETL_SCRIPT="${SCRIPT_DIR}/run_sgmi_etl_automated.sh"
PYTHON_SCRIPT="${SCRIPT_DIR}/sgmi_view_builder.py"

# Verify scripts exist
if [ ! -f "${PIPELINE_SCRIPT}" ]; then
    echo "Error: Pipeline script not found at ${PIPELINE_SCRIPT}"
    exit 1
fi
if [ ! -f "${ETL_SCRIPT}" ]; then
    echo "Error: ETL script not found at ${ETL_SCRIPT}"
    exit 1
fi

echo "Executing pipeline inside Docker container on brev_default network..."
echo "Mounting: ${REPO_ABS} -> /workspace"
echo ""

# Data and scripts are now part of extract service, accessible via /workspace mount
# Users can add files to /home/aModels/services/extract/data/training/sgmi and they'll be accessible
DATA_DIR="${REPO_ABS}/services/extract/data/training/sgmi"
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# Read shell script contents for embedding (before Docker command)
PIPELINE_SCRIPT_CONTENT=$(cat "${PIPELINE_SCRIPT}")
ETL_SCRIPT_CONTENT=$(cat "${ETL_SCRIPT}")

# Read Python script and encode it for safe embedding (before Docker command)
PYTHON_SCRIPT_B64=""
if [ -f "${PYTHON_SCRIPT}" ]; then
    PYTHON_SCRIPT_B64=$(base64 -w 0 < "${PYTHON_SCRIPT}" 2>/dev/null || base64 < "${PYTHON_SCRIPT}" 2>/dev/null | tr -d '\n')
fi

# Run pipeline in container - data accessible via workspace mount, scripts embedded
docker run --rm \
    --network brev_default \
    -v "${REPO_ABS}:/workspace:ro" \
    -w /workspace/services/extract \
    -e EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL}" \
    -e CATALOG_SERVICE_URL="${CATALOG_SERVICE_URL}" \
    -e SGMI_DATA_ROOT="/workspace/services/extract/data" \
    -e PYTHON_SCRIPT_B64="${PYTHON_SCRIPT_B64}" \
    alpine:latest \
    sh -c "
        apk add --no-cache bash curl python3 py3-pip >/dev/null 2>&1
        pip3 install requests simple-ddl-parser sqlglot >/dev/null 2>&1 || true
        cd /workspace/services/extract
        mkdir -p /tmp/scripts/pipelines /tmp/logs/sgmi_pipeline
        # Extract Python script from base64
        if [ -n \"\${PYTHON_SCRIPT_B64}\" ]; then
            echo \"\${PYTHON_SCRIPT_B64}\" | base64 -d > /tmp/scripts/pipelines/sgmi_view_builder.py 2>&1
            chmod +x /tmp/scripts/pipelines/sgmi_view_builder.py
            echo 'Python script extracted from base64' >&2
        else
            echo 'ERROR: Python script base64 not provided' >&2
            exit 1
        fi
        # Copy shell scripts into container using heredoc with embedded content
        cat > /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh <<PIPELINE_EOF
${PIPELINE_SCRIPT_CONTENT}
PIPELINE_EOF
        cat > /tmp/scripts/pipelines/run_sgmi_etl_automated.sh <<ETL_EOF
${ETL_SCRIPT_CONTENT}
ETL_EOF
        chmod +x /tmp/scripts/pipelines/*.sh
        # Update paths in scripts to point to correct locations
        sed -i 's|SCRIPT_DIR=.*|SCRIPT_DIR=/tmp/scripts/pipelines|' /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh
        sed -i 's|REPO_ROOT=.*|REPO_ROOT=/workspace|' /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh
        sed -i 's|LOG_DIR=.*|LOG_DIR=/tmp/logs/sgmi_pipeline|' /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh
        sed -i 's|REPO_ROOT=.*|REPO_ROOT=/workspace|' /tmp/scripts/pipelines/run_sgmi_etl_automated.sh
        sed -i 's|LOG_DIR=.*|LOG_DIR=/tmp/logs/sgmi_pipeline|' /tmp/scripts/pipelines/run_sgmi_etl_automated.sh
        # Verify data is accessible via workspace mount
        echo 'Verifying data access at /workspace/services/extract/data/training/sgmi...'
        ls -la /workspace/services/extract/data/training/sgmi/ 2>&1 | head -10 || echo 'Data directory check failed'
        bash /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh
    "
