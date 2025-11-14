# Multi-stage build for Postgres Lang Service

# Stage 1: Build
FROM golang:1.24-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git gcc musl-dev

WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the server binary
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o /build/server ./cmd/server

# Build healthcheck binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o /build/healthcheck ./cmd/healthcheck

# Stage 2: Runtime
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN addgroup -g 1000 postgres && \
    adduser -D -u 1000 -G postgres postgres

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /build/server /app/server
COPY --from=builder /build/healthcheck /app/healthcheck

# Copy migrations (optional, for reference)
COPY --from=builder /build/migrations /app/migrations

# Change ownership
RUN chown -R postgres:postgres /app

# Switch to non-root user
USER postgres

# Expose gRPC port and Flight port
EXPOSE 50055 8825

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/app/healthcheck", "-addr=localhost:50055", "-timeout=2s"]

# Run the server
ENTRYPOINT ["/app/server"]
