# syntax=docker/dockerfile:1.6
##
## graph-server builder
##
FROM golang:1.24-alpine AS builder

WORKDIR /workspace

RUN apk add --no-cache git build-base sqlite-dev

# Copy the entire mono-repo to preserve relative paths for replace directives
COPY . /workspace/src

WORKDIR /workspace/src/services/graph

# Use the image toolchain (Go 1.24+)
ENV GOTOOLCHAIN=local

# Optional: Remove go.work file if it causes conflicts with Docker build
# The go.mod file now has clean, explicit replace directives
RUN rm -f ../../go.work || true

# Verify go.mod is valid and download dependencies
# No sed hacks needed - go.mod is now clean and explicit
RUN go mod download
RUN go mod verify

# Build with CGO disabled for portability (except SQLite which needs CGO)
# SQLite requires CGO for mattn/go-sqlite3
RUN CGO_ENABLED=1 go build -o /workspace/build/graph-server ./cmd/graph-server

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
