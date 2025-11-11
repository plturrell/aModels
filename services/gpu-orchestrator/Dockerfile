# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o gpu-orchestrator ./main.go

# Runtime stage
FROM alpine:latest

# Install nvidia-smi dependencies (for GPU monitoring)
# Note: nvidia-smi should be available from host via volume mount
# This just ensures the container can run it
RUN apk --no-cache add ca-certificates curl

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/gpu-orchestrator .

# Expose port
EXPOSE 8088

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8088/healthz || exit 1

# Run the service
CMD ["./gpu-orchestrator"]

