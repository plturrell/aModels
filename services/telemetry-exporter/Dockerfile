FROM golang:1.24-alpine AS builder

WORKDIR /workspace

# Install git (required for go mod download with replace directives)
RUN apk add --no-cache git

# Copy pkg/localai and pkg/catalog FIRST (required by testing service)
# The replace directive in testing/go.mod will use these local copies
# instead of the broken published version
COPY pkg/localai/ pkg/localai/
COPY pkg/catalog/ pkg/catalog/

# Copy testing service (required by replace directive) - must match replace path
# This must come after pkg/localai so the replace directive can find it
COPY services/testing/ services/testing/

# Copy telemetry-exporter service
COPY services/telemetry-exporter/go.mod services/telemetry-exporter/go.sum* services/telemetry-exporter/
COPY services/telemetry-exporter/ services/telemetry-exporter/

# Download dependencies and generate go.sum
# The replace directives in both go.mod files will use local pkg/localai instead of broken published version
WORKDIR /workspace/services/telemetry-exporter
RUN go mod download && go mod tidy

# Build the application WITH LocalAI support (no nolocalai tag)
RUN CGO_ENABLED=0 GOOS=linux go build -o /workspace/server ./cmd/server

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates curl

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /workspace/server ./server

# Create data directory
RUN mkdir -p /app/data/traces

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["./server"]

