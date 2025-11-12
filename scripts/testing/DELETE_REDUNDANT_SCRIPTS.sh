#!/bin/bash
# Script to delete redundant test scripts after consolidation
# Run this ONLY after verifying the enhanced scripts work correctly
#
# This will delete 9 redundant scripts from /testing/ and replace 2 old versions

set -e

echo "=== REDUNDANT SCRIPT DELETION TOOL ==="
echo ""
echo "⚠️  WARNING: This will permanently delete 11 files!"
echo ""
echo "Files to be deleted from /home/aModels/testing/:"
echo "  - run_all_tests_final.sh"
echo "  - run_all_tests_fixed.sh"
echo "  - run_all_tests_with_step0.sh"
echo "  - run_all_tests_working.sh"
echo "  - run_tests_docker_network.sh"
echo "  - run_tests_from_container.sh"
echo "  - run_tests_from_docker.sh"
echo "  - run_tests_in_container.sh"
echo "  - run_tests_now.sh"
echo ""
echo "Files to be replaced:"
echo "  - run_all_tests.sh (old) → enhanced version in scripts/testing/"
echo "  - run_tests_docker.sh (old) → enhanced version in scripts/testing/"
echo ""
echo "Enhanced versions are in: /home/aModels/scripts/testing/"
echo ""

read -p "Are you sure you want to proceed? (type 'DELETE' to confirm): " confirmation

if [ "$confirmation" != "DELETE" ]; then
    echo "❌ Deletion cancelled"
    exit 1
fi

echo ""
echo "Deleting redundant scripts..."
echo ""

cd /home/aModels/testing

# Delete redundant variants
rm -v run_all_tests_final.sh 2>/dev/null || echo "  (already removed: run_all_tests_final.sh)"
rm -v run_all_tests_fixed.sh 2>/dev/null || echo "  (already removed: run_all_tests_fixed.sh)"
rm -v run_all_tests_with_step0.sh 2>/dev/null || echo "  (already removed: run_all_tests_with_step0.sh)"
rm -v run_all_tests_working.sh 2>/dev/null || echo "  (already removed: run_all_tests_working.sh)"
rm -v run_tests_docker_network.sh 2>/dev/null || echo "  (already removed: run_tests_docker_network.sh)"
rm -v run_tests_from_container.sh 2>/dev/null || echo "  (already removed: run_tests_from_container.sh)"
rm -v run_tests_from_docker.sh 2>/dev/null || echo "  (already removed: run_tests_from_docker.sh)"
rm -v run_tests_in_container.sh 2>/dev/null || echo "  (already removed: run_tests_in_container.sh)"
rm -v run_tests_now.sh 2>/dev/null || echo "  (already removed: run_tests_now.sh)"

# Delete old versions (replaced by enhanced)
rm -v run_all_tests.sh 2>/dev/null || echo "  (already removed: run_all_tests.sh)"
rm -v run_tests_docker.sh 2>/dev/null || echo "  (already removed: run_tests_docker.sh)"

echo ""
echo "✓ Deletion complete"
echo ""
echo "Remaining files in /home/aModels/testing/:"
ls -1 /home/aModels/testing/*.sh 2>/dev/null || echo "  (no .sh files remaining)"
echo ""
echo "Enhanced scripts available in /home/aModels/scripts/testing/:"
ls -1 /home/aModels/scripts/testing/*.sh
echo ""
echo "✅ Consolidation complete - use scripts in /scripts/testing/ going forward"
