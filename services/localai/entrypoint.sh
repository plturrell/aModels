#!/bin/bash
set -e

# Remove domains.json if it's a directory (from previous mount issues)
if [ -d "/workspace/config/domains.json" ]; then
    echo "⚠️  Removing /workspace/config/domains.json directory (should be a file)"
    rm -rf /workspace/config/domains.json
fi

# Ensure config directory exists
mkdir -p /workspace/config

# Get config path from environment or use default
CONFIG_PATH="${DOMAIN_CONFIG_PATH:-/workspace/config/domains.json}"

# Run the server
exec /usr/local/bin/vaultgemma-server -config "$CONFIG_PATH" -port 8080

