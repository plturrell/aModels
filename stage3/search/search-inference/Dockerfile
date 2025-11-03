# syntax=docker/dockerfile:1.6
FROM golang:1.22.6-alpine3.20 AS builder

ARG UPSTREAM_REPO=https://github.com/plturrell/agenticAiETH.git

WORKDIR /workspace

RUN apk add --no-cache git build-base

COPY . /workspace/export
RUN git clone --depth 1 "${UPSTREAM_REPO}" /workspace/upstream

RUN mkdir -p /workspace/src && \
    cp -R /workspace/export/stage3/search/search-inference /workspace/src/search-inference && \
    cp -R /workspace/upstream/agenticAiETH_layer4_AgentSDK /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_LocalAI /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_HANA /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_Extract /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_Models /workspace/src/

WORKDIR /workspace/src/search-inference

ENV GOTOOLCHAIN=go1.22.6

RUN go mod tidy
RUN CGO_ENABLED=0 go build -o /workspace/build/search-inference ./cmd/search-inference

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /workspace

COPY --from=builder /workspace/build/search-inference /usr/local/bin/search-inference

EXPOSE 8090

ENTRYPOINT ["/usr/local/bin/search-inference"]
