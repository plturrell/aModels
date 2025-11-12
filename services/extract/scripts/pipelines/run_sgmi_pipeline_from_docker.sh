#!/usr/bin/env bash
# Run SGMI pipeline from within Docker network
# This script runs the pipeline inside a container on the same network as services
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

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

# Copy script content into container and execute
# This avoids the volume mount issue with the scripts directory
# Stream data files into container using tar to work around overlay filesystem issue
# Users can add files to /home/aModels/data/training/sgmi and they'll be automatically included
DATA_DIR="${REPO_ABS}/data/training/sgmi"
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# Read Python script content and encode it for safe embedding (before Docker command)
PYTHON_SCRIPT_B64=""
if [ -f "${PYTHON_SCRIPT}" ]; then
    PYTHON_SCRIPT_B64=$(base64 -w 0 < "${PYTHON_SCRIPT}" 2>/dev/null || base64 < "${PYTHON_SCRIPT}" 2>/dev/null | tr -d '\n')
fi

# Stream data into container using tar (works around overlay filesystem masking)
# Also include Python script in a separate tar stream
docker run --rm \
    --network brev_default \
    -v "${REPO_ABS}:/workspace:ro" \
    -w /workspace/services/extract \
    -i \
    -e EXTRACT_SERVICE_URL="${EXTRACT_SERVICE_URL}" \
    -e DMS_SERVICE_URL="${DMS_SERVICE_URL}" \
    -e CATALOG_SERVICE_URL="${CATALOG_SERVICE_URL}" \
    -e SGMI_DATA_ROOT="/sgmi-data" \
    -e PYTHON_SCRIPT_B64="${PYTHON_SCRIPT_B64}" \
    alpine:latest \
    sh -c "
        apk add --no-cache bash curl python3 py3-pip tar >/dev/null 2>&1
        pip3 install requests simple-ddl-parser sqlglot >/dev/null 2>&1 || true
        cd /workspace/services/extract
        # Extract combined tar archive containing both data and Python script
        echo 'Extracting SGMI data and Python script...'
        mkdir -p /sgmi-data/training /tmp/scripts/pipelines
        # Save tar stream to temp file so we can extract multiple times
        TMP_STREAM=\$(mktemp)
        cat > "\${TMP_STREAM}"
        # Extract both data and Python script in one pass
        # Extract data to /sgmi-data/training
        tar -xzf "\${TMP_STREAM}" -C /sgmi-data/training sgmi 2>/dev/null || echo 'Note: Data extraction completed'
        # Extract Python script to /tmp
        tar -xzf "\${TMP_STREAM}" -C /tmp sgmi_view_builder.py 2>&1 | head -3 >&2 || true
        # Move Python script to correct location if extracted
        if [ -f /tmp/sgmi_view_builder.py ]; then
            mv /tmp/sgmi_view_builder.py /tmp/scripts/pipelines/ 2>/dev/null
            echo 'Python script extracted from tar successfully' >&2
        elif [ -n "\${PYTHON_SCRIPT_B64}" ]; then
            echo 'Python script not in tar, using base64 fallback...' >&2
            echo "\${PYTHON_SCRIPT_B64}" | base64 -d > /tmp/scripts/pipelines/sgmi_view_builder.py 2>&1
        else
            echo 'ERROR: Python script extraction failed and no base64 fallback' >&2
            echo 'Checking tar contents...' >&2
            tar -tzf "\${TMP_STREAM}" 2>&1 | grep -E 'sgmi_view|sgmi/' | head -5 >&2
        fi
        rm -f "\${TMP_STREAM}"
        # Copy both shell scripts into container using heredoc
        cat > /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh <<'PIPELINE_EOF'
$(cat "${PIPELINE_SCRIPT}")
PIPELINE_EOF
        cat > /tmp/scripts/pipelines/run_sgmi_etl_automated.sh <<'ETL_EOF'
$(cat "${ETL_SCRIPT}")
ETL_EOF
        # Verify Python script exists
        if [ ! -f /tmp/scripts/pipelines/sgmi_view_builder.py ]; then
            echo 'ERROR: Python script was not created successfully' >&2
            echo 'Files in /tmp/scripts/pipelines/: ' >&2
            ls -la /tmp/scripts/pipelines/ >&2
            exit 1
        fi
        # Verify Python script has content
        if [ ! -f /tmp/scripts/pipelines/sgmi_view_builder.py ]; then
            echo 'ERROR: Python script file does not exist' >&2
            exit 1
        fi
        if [ ! -s /tmp/scripts/pipelines/sgmi_view_builder.py ]; then
            echo 'ERROR: Python script file is empty' >&2
            exit 1
        fi
        # Verify file has content (already checked with -s above)
        # Get line count for logging
        script_lines=$(wc -l < /tmp/scripts/pipelines/sgmi_view_builder.py 2>/dev/null | tr -d ' ')
        file_info=$(ls -lh /tmp/scripts/pipelines/sgmi_view_builder.py 2>/dev/null | awk '{print $5}')
        if [ -z "\${script_lines}" ]; then
            script_lines=unknown
        fi
        if [ -z "\${file_info}" ]; then
            file_info=unknown
        fi
        echo "Python script verified: \${script_lines} lines, \${file_info}" >&2
        chmod +x /tmp/scripts/pipelines/*.sh
        # Update SCRIPT_DIR and REPO_ROOT in the pipeline script to point to correct locations
        sed -i 's|SCRIPT_DIR=.*|SCRIPT_DIR=/tmp/scripts/pipelines|' /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh
        sed -i 's|REPO_ROOT=.*|REPO_ROOT=/workspace|' /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh
        sed -i 's|LOG_DIR=.*|LOG_DIR=/tmp/logs/sgmi_pipeline|' /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh
        sed -i 's|REPO_ROOT=.*|REPO_ROOT=/workspace|' /tmp/scripts/pipelines/run_sgmi_etl_automated.sh
        sed -i 's|LOG_DIR=.*|LOG_DIR=/tmp/logs/sgmi_pipeline|' /tmp/scripts/pipelines/run_sgmi_etl_automated.sh
        mkdir -p /tmp/logs/sgmi_pipeline
        # Verify data is accessible
        echo 'Verifying data access at /sgmi-data/training/sgmi...'
        ls -la /sgmi-data/training/sgmi/ 2>&1 | head -10 || echo 'Data directory check failed'
        bash /tmp/scripts/pipelines/run_sgmi_complete_pipeline.sh
    " < <(
        # Create a combined tar archive with both data and Python script
        # Use uncompressed tar first, then compress
        TMP_TAR=$(mktemp)
        # Add SGMI data
        cd "${REPO_ABS}/data/training" && tar -cf "${TMP_TAR}" sgmi 2>/dev/null
        # Append Python script (uncompressed tar allows append)
        if [ -f "${PYTHON_SCRIPT}" ]; then
            cd "$(dirname "${PYTHON_SCRIPT}")" && tar --append -f "${TMP_TAR}" "$(basename "${PYTHON_SCRIPT}")" 2>/dev/null
        fi
        # Compress and output
        gzip -c "${TMP_TAR}" 2>/dev/null
        rm -f "${TMP_TAR}" 2>/dev/null
    )
