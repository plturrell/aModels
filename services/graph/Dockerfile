# syntax=docker/dockerfile:1.6
##
## graph-server builder
##
FROM golang:1.24-alpine AS builder

WORKDIR /workspace

RUN apk add --no-cache git build-base

# Copy the entire repo - agenticAiETH dependencies removed, build will fail if dependencies are missing
COPY . /workspace/src

# NOTE: This service requires agenticAiETH mono-repo dependencies that are not available:
# - agenticAiETH_layer1_Blockchain
# - agenticAiETH_layer4_AgentSDK  
# - agenticAiETH_layer4_Extract
# - agenticAiETH_layer4_HANA
# - agenticAiETH_layer4_Postgres
# - agenticAiETH_layer4_Models/maths
# - third_party/go-redis, third_party/go-arrow
#
# To build this service, you need to either:
# 1. Provide these dependencies in the expected paths
# 2. Update go.mod to remove/update replace directives
# 3. Refactor code to remove dependencies

WORKDIR /workspace/src/services/graph

# Use the image toolchain (Go 1.25)
ENV GOTOOLCHAIN=local

# Remove go.work file if present to avoid version conflicts
RUN rm -f ../../go.work || true

# Remove replace directives that point to missing local paths
# Keep Postgres replace but fix path
RUN sed -i '/replace.*third_party\/go-arrow/d' go.mod || true && \
    sed -i '/replace.*third_party\/go-redis/d' go.mod || true && \
    sed -i '/replace.*infrastructure\/third_party\/go-hdb/d' go.mod || true && \
    sed -i '/replace.*agenticAiETH_layer1_Blockchain/d' go.mod || true && \
    sed -i '/replace.*agenticAiETH_layer4_HANA/d' go.mod || true && \
    sed -i '/replace.*agenticAiETH_layer4_AgentSDK/d' go.mod || true && \
    sed -i '/replace.*agenticAiETH_layer4_Extract/d' go.mod || true && \
    sed -i '/replace.*agenticAiETH_layer4_Models/d' go.mod || true && \
    sed -i '/replace.*agenticAiETH_layer4_Postgres/d' go.mod || true

# Add replace directives using go mod edit (must be done before go mod tidy)
# Remove postgres from require if present, then add replace directives
RUN sed -i '/^require/,/^)/ { /agenticAiETH_layer4_Postgres/d; }' go.mod && \
    go mod edit -replace github.com/plturrell/aModels/services/graph=. && \
    go mod edit -replace github.com/SAP/go-hdb=../../infrastructure/third_party/go-hdb && \
    go mod edit -replace github.com/plturrell/aModels/services/postgres=../postgres && \
    go mod edit -replace github.com/plturrell/aModels/services/catalog=../catalog && \
    go mod edit -replace github.com/plturrell/aModels/services/extract=../extract && \
    go mod edit -replace github.com/plturrell/aModels/services/orchestration=../orchestration && \
    go mod edit -replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres=../postgres && \
    go mod edit -replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Extract/gen/extractpb=../extract/gen/extractpb

RUN go mod tidy
RUN CGO_ENABLED=0 go build -o /workspace/build/graph-server ./cmd/graph-server

##
## runtime image (CUDA friendly base)
##
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /workspace

COPY --from=builder /workspace/build/graph-server /usr/local/bin/graph-server

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/graph-server"]
