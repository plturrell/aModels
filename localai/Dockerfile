# syntax=docker/dockerfile:1.6
##
## LocalAI builder (VaultGemma server)
##
FROM golang:1.24-alpine AS builder

WORKDIR /workspace

RUN apk add --no-cache git build-base

# Copy the entire repo to maintain the directory structure needed for go.mod replace directives
COPY . /workspace/src

# Setup the directory structure expected by go.mod replace directives
RUN mkdir -p /workspace/src/models && \
    mkdir -p /workspace/src/third_party && \
    if [ ! -d /workspace/src/third_party/go-llama.cpp ]; then \
        echo "Warning: third_party/go-llama.cpp not found, may need to be provided" ; \
    fi

WORKDIR /workspace/src/localai

ENV GOTOOLCHAIN=local
ENV GOWORK=off
RUN go mod download
RUN CGO_ENABLED=0 go build -mod=mod -o /workspace/build/vaultgemma ./cmd/vaultgemma-server

##
## runtime image
##
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MODELS_DIR=/models

WORKDIR /workspace

COPY --from=builder /workspace/build/vaultgemma /usr/local/bin/vaultgemma-server

EXPOSE 8081

ENTRYPOINT ["/usr/local/bin/vaultgemma-server"]
