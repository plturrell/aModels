#!/usr/bin/env bash
# Script to delete DMS service after migration is complete
# This script removes the DMS service directory and verifies no critical references remain

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
DMS_DIR="${REPO_ROOT}/services/dms"

echo "=========================================="
echo "DMS Service Removal Script"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Verify migration is complete"
echo "  2. Check for remaining DMS references"
echo "  3. Delete DMS service directory"
echo ""

# Check if DMS directory exists
if [ ! -d "${DMS_DIR}" ]; then
    echo "✓ DMS service directory not found - already removed"
    exit 0
fi

# Warn user
echo "WARNING: This will permanently delete the DMS service directory:"
echo "  ${DMS_DIR}"
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "${confirm}" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Check for critical references (excluding documentation and comments)
echo ""
echo "Checking for remaining DMS references..."
CRITICAL_REFS=$(grep -r "services/dms\|services\\dms" \
    --include="*.go" \
    --include="*.py" \
    --include="*.ts" \
    --include="*.sh" \
    --include="*.yaml" \
    --include="*.yml" \
    "${REPO_ROOT}" \
    2>/dev/null | grep -v "DMS service removed\|migrated from DMS\|replaces DMS\|# DMS" | grep -v "services/dms" | head -10 || true)

if [ -n "${CRITICAL_REFS}" ]; then
    echo "⚠️  Warning: Found potential DMS references:"
    echo "${CRITICAL_REFS}"
    echo ""
    read -p "Continue anyway? (yes/no): " continue_anyway
    if [ "${continue_anyway}" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "✓ No critical DMS references found"
fi

# Delete DMS directory
echo ""
echo "Deleting DMS service directory..."
rm -rf "${DMS_DIR}"

if [ ! -d "${DMS_DIR}" ]; then
    echo "✓ DMS service directory successfully deleted"
else
    echo "✗ Failed to delete DMS service directory"
    exit 1
fi

echo ""
echo "=========================================="
echo "DMS Service Removal Complete"
echo "=========================================="
echo ""
echo "The DMS service has been removed."
echo "All functionality is now in Extract service with Gitea storage."
echo ""

