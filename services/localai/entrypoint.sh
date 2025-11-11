#!/bin/bash
set -e

# Get config path from environment or use default
CONFIG_PATH="${DOMAIN_CONFIG_PATH:-/workspace/config/domains.json}"

# If domains.json is a directory, try to find the actual file
if [ -d "/workspace/config/domains.json" ]; then
    echo "⚠️  /workspace/config/domains.json is a directory, checking for actual file..."
    # Check if there's a file inside the directory
    if [ -f "/workspace/config/domains.json/domains.json" ]; then
        CONFIG_PATH="/workspace/config/domains.json/domains.json"
        echo "✅ Found domains.json inside directory: $CONFIG_PATH"
    # Check if there's a file at the parent level
    elif [ -f "/workspace/config/domains.json" ]; then
        CONFIG_PATH="/workspace/config/domains.json"
        echo "✅ Found domains.json file: $CONFIG_PATH"
    else
        echo "⚠️  Could not find domains.json file, checking mounted config directory..."
        # List all JSON files in config directory
        find /workspace/config -name "*.json" -type f 2>/dev/null | head -5
        echo "⚠️  Using default path: $CONFIG_PATH"
    fi
fi

# Run the server
exec /usr/local/bin/vaultgemma-server -config "$CONFIG_PATH" -port 8080

