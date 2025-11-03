# syntax=docker/dockerfile:1.6
##
## LocalAI builder (VaultGemma server)
##
FROM golang:1.22.6-alpine3.20 AS builder

ARG UPSTREAM_REPO=https://github.com/plturrell/agenticAiETH.git

WORKDIR /workspace

RUN apk add --no-cache git build-base

COPY . /workspace/export
RUN git clone --depth 1 "${UPSTREAM_REPO}" /workspace/upstream

RUN mkdir -p /workspace/src && \
    cp -R /workspace/export/localai /workspace/src/localai && \
    cp -R /workspace/upstream/agenticAiETH_layer1_Blockchain /workspace/src/ && \
    cp -R /workspace/upstream/agenticAiETH_layer4_AgentSDK /workspace/src/ && \
    mkdir -p /workspace/src/agenticAiETH_layer4_Training/models && \
    cp -R /workspace/export/models/glove /workspace/src/agenticAiETH_layer4_Training/models/glove && \
    cp -R /workspace/export/models/sentencepiece /workspace/src/agenticAiETH_layer4_Training/models/sentencepiece && \
    mkdir -p /workspace/src/agenticAiETH_layer4_Models && \
    cp -R /workspace/upstream/agenticAiETH_layer4_Models/maths /workspace/src/agenticAiETH_layer4_Models/ && \
    mkdir -p /workspace/src/third_party && \
    cp -R /workspace/upstream/third_party/go-llama.cpp /workspace/src/third_party/ && \
    cp -R /workspace/upstream/third_party/go-arrow /workspace/src/third_party/

WORKDIR /workspace/src/localai

ENV GOTOOLCHAIN=go1.22.6

RUN go mod tidy
RUN CGO_ENABLED=0 go build -o /workspace/build/vaultgemma ./cmd/vaultgemma-server

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
