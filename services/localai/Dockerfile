# syntax=docker/dockerfile:1.6
##
## LocalAI builder (VaultGemma server)
##
FROM golang:1.22-alpine AS builder

WORKDIR /workspace

RUN apk add --no-cache git build-base

# Copy LocalAI source and the local model modules required by go.mod replaces
COPY services/localai/ /workspace/services/localai/
COPY models/glove/ /workspace/services/models/glove/
COPY models/sentencepiece/ /workspace/services/models/sentencepiece/

WORKDIR /workspace/services/localai

ENV GOTOOLCHAIN=local
ENV GOWORK=off

RUN go mod download
RUN CGO_ENABLED=0 go build -mod=mod -o /workspace/build/vaultgemma ./cmd/vaultgemma-server

##
## runtime image
##
FROM ubuntu:22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MODELS_DIR=/models

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/build/vaultgemma /usr/local/bin/vaultgemma-server

# Copy entrypoint script
COPY services/localai/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Config directory will be mounted via volume in docker-compose
RUN mkdir -p /workspace/config

WORKDIR /workspace

EXPOSE 8081

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
