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

# Use the image toolchain (Go 1.24)
ENV GOTOOLCHAIN=local

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
