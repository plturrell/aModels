#!/usr/bin/env bash
# dev.sh – reproducible Go 1.24 build wrapper
set -e

IMAGE="orchestration-dev"

echo "Building dev container..."
docker build -t "$IMAGE" -f Dockerfile.dev . >/dev/null

echo "Running tests in Go 1.24..."
exec docker run --rm -e GOWORK=off -v "$(pwd)":/workspace "$IMAGE" "$@"
