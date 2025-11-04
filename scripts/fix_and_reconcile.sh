#!/usr/bin/env bash
# Complete fix: Enrich properties + Reconcile graph
set -euo pipefail

echo "=== Complete Fix: Properties + Reconciliation ==="
echo ""

# Step 1: Enrich missing properties
echo "Step 1: Enriching missing properties..."
./scripts/enrich_missing_properties.sh
echo ""

# Step 2: Check reconciliation
echo "Step 2: Checking graph reconciliation..."
./scripts/reconcile_graph_to_postgres.sh
echo ""

# Step 3: Verify quality metrics
echo "Step 3: Verifying quality metrics..."
./scripts/run_quality_metrics.sh

