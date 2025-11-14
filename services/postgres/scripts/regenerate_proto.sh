#!/bin/bash
# Regenerate protobuf files for postgres service
# This script uses Docker to ensure consistent tool versions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Regenerating protobuf files for postgres service..."

# Use Docker to regenerate with consistent versions
# We use protoc v5.28.3 (compatible with protoc-gen-go v1.36.10)
docker run --rm \
  -v "$SERVICE_DIR:/workspace" \
  -w /workspace \
  golang:1.24-alpine \
  sh -c "
    apk add --no-cache curl unzip > /dev/null 2>&1 && \
    curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v28.3/protoc-28.3-linux-x86_64.zip > /dev/null 2>&1 && \
    unzip -q protoc-28.3-linux-x86_64.zip -d /tmp/protoc && \
    chmod +x /tmp/protoc/bin/protoc && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.10 > /dev/null 2>&1 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest > /dev/null 2>&1 && \
    export PATH=\$PATH:/go/bin:/tmp/protoc/bin && \
    cat proto/v1/postgres_lang_service.proto | \
    sh -c 'mkdir -p /tmp/proto/v1 && cat > /tmp/proto/v1/postgres_lang_service.proto' && \
    /tmp/protoc/bin/protoc \
      --go_out=pkg/gen/v1 \
      --go_opt=paths=source_relative \
      --go-grpc_out=pkg/gen/v1 \
      --go-grpc_opt=paths=source_relative \
      --proto_path=/tmp/proto/v1 \
      --proto_path=/tmp/protoc/include \
      /tmp/proto/v1/postgres_lang_service.proto && \
    echo 'Successfully regenerated protobuf files' && \
    ls -lh pkg/gen/v1/*.pb.go && \
    head -5 pkg/gen/v1/postgres_lang_service.pb.go
  "

echo "Protobuf files regenerated successfully!"

