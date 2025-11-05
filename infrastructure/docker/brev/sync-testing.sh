#!/bin/bash
# Sync testing files from host to training-shell container
# Usage: ./sync-testing.sh

set -e

CONTAINER_NAME="training-shell"
HOST_TESTING_DIR="/home/aModels/testing"
CONTAINER_TESTING_DIR="/workspace/testing"

echo "Syncing testing files to container..."
echo "  From: $HOST_TESTING_DIR"
echo "  To: $CONTAINER_NAME:$CONTAINER_TESTING_DIR"

if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ Container $CONTAINER_NAME is not running"
    echo "   Start it with: docker compose -f infrastructure/docker/brev/docker-compose.yml up -d trainer"
    exit 1
fi

docker cp "$HOST_TESTING_DIR/." "${CONTAINER_NAME}:${CONTAINER_TESTING_DIR}/"

FILE_COUNT=$(docker exec "$CONTAINER_NAME" find "$CONTAINER_TESTING_DIR" -type f | wc -l)
echo "✅ Files synced successfully"
echo "   Found $FILE_COUNT files in container"

