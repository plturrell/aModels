#!/bin/bash

# BCBS239 Audit Server - Quick Start Script

set -e

echo "🏦 BCBS239 Audit Server - Starting..."
echo ""

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go first."
    exit 1
fi

# Navigate to regulatory service directory
cd "$(dirname "$0")"

echo "📦 Building audit server..."
go build -o audit-server ./cmd/audit-server

echo "✅ Build complete!"
echo ""
echo "🚀 Starting BCBS239 Audit Server..."
echo ""
echo "   📊 UI: http://localhost:8099"
echo "   🔌 API: http://localhost:8099/api/compliance/audit/"
echo "   ❤️  Health: http://localhost:8099/healthz"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Set default environment variables if not already set
export NEO4J_URL=${NEO4J_URL:-"bolt://localhost:7687"}
export NEO4J_USER=${NEO4J_USER:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"password"}
export LOCALAI_URL=${LOCALAI_URL:-"http://localhost:8080"}
export GNN_SERVICE_URL=${GNN_SERVICE_URL:-"http://localhost:8081"}
export GOOSE_SERVER_URL=${GOOSE_SERVER_URL:-"http://localhost:8082"}
export DEEPAGENTS_URL=${DEEPAGENTS_URL:-"http://localhost:8083"}
export AUDIT_SERVER_ADDR=${AUDIT_SERVER_ADDR:-":8099"}

# Run the server
./audit-server
