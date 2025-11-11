#!/bin/bash
set -e

# Production startup script for LocalAI VaultGemma Server
# Uses offline-only config with local safetensors models

cd "$(dirname "$0")"

# Ensure bin directory exists
mkdir -p bin logs

# Build the server if binary doesn't exist
if [ ! -f "bin/vaultgemma-server" ]; then
    echo "🔨 Building vaultgemma-server..."
    PATH=/usr/local/go/bin:$PATH GO111MODULE=on GOWORK=off \
        go build -o bin/vaultgemma-server ./cmd/vaultgemma-server
    echo "✅ Build complete"
fi

# Stop any existing server
echo "🛑 Stopping existing server..."
pkill -f vaultgemma-server || true
sleep 2

# Start server with production config
echo "🚀 Starting VaultGemma server with production config..."
DOMAIN_CONFIG_PATH=config/domains.production.json \
    nohup ./bin/vaultgemma-server \
        -config config/domains.production.json \
        -model ../../models/vaultgemma-1b-transformers \
        -port 8080 \
        > logs/vaultgemma-server.log 2>&1 &

# Wait for server to start
sleep 3

# Check if server is running
if ps aux | grep -q "[v]aultgemma-server"; then
    echo "✅ Server started successfully"
    echo "📋 Logs: tail -f logs/vaultgemma-server.log"
    echo "🌐 Health: curl http://localhost:8080/health"
    echo "📊 Models: curl http://localhost:8080/v1/models"
    echo "💬 Chat: curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"vaultgemma\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"
else
    echo "❌ Server failed to start. Check logs/vaultgemma-server.log"
    exit 1
fi


