# syntax=docker/dockerfile:1.6
##
## graph-server builder
##
FROM golang:1.22.6-alpine3.20 AS builder

ARG UPSTREAM_REPO=https://github.com/plturrell/agenticAiETH.git

WORKDIR /workspace

RUN apk add --no-cache git build-base

# Copy the exported stack (this repo) into the image.
COPY . /workspace/export

# Pull the upstream mono-repo so we can satisfy go.mod replace directives.
RUN git clone --depth 1 "${UPSTREAM_REPO}" /workspace/upstream

# Mirror the directory structure expected by go.mod in the builder workspace.
RUN mkdir -p /workspace/src && \
    cp -R /workspace/export/stage3/graph /workspace/src/agenticAiETH_layer4_Graph && \
    cp -R /workspace/upstream/agenticAiETH_layer1_Blockchain /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_AgentSDK /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_Extract /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_HANA /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_Postgres /workspace/src/ && \
    mkdir -p /workspace/src/agenticAiETH_layer4_Models && \
    cp -R /workspace/upstream/agenticAiETH_layer4_Models/maths /workspace/src/agenticAiETH_layer4_Models/ && \
    mkdir -p /workspace/src/third_party && \
    cp -R /workspace/upstream/third_party/go-redis /workspace/src/third_party/ && \
    cp -R /workspace/upstream/third_party/go-arrow /workspace/src/third_party/

WORKDIR /workspace/src/agenticAiETH_layer4_Graph

# Force a toolchain that exists today even though go.mod points at 1.25.x.
ENV GOTOOLCHAIN=go1.22.6

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
