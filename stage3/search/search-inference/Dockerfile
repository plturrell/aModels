# syntax=docker/dockerfile:1.6
FROM golang:1.23-alpine AS builder

WORKDIR /workspace

# SQLite requires CGO, so we need gcc and musl-dev
RUN apk add --no-cache git build-base gcc musl-dev sqlite-dev

# Copy the entire repo
COPY . /workspace/src

WORKDIR /workspace/src/stage3/search/search-inference

ENV GOTOOLCHAIN=local
ENV CGO_ENABLED=1
ENV GOWORK=off

# SQLite needs CGO enabled
RUN go mod tidy
RUN cd cmd/search-server && go build -o /workspace/build/search-inference .

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /workspace

# Install SQLite runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/build/search-inference /usr/local/bin/search-inference

EXPOSE 8090

ENTRYPOINT ["/usr/local/bin/search-inference"]
