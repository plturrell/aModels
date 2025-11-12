FROM golang:1.24-alpine AS builder

WORKDIR /workspace

# Copy root go.mod if it exists (for replace directive)
COPY go.mod* ./

# Copy all required modules for go.mod replace directives
COPY services/shared ./services/shared
COPY services/postgres ./services/postgres
COPY infrastructure/third_party/sql-parser-go ./infrastructure/third_party/sql-parser-go
COPY infrastructure/third_party/go-hdb ./infrastructure/third_party/go-hdb
COPY infrastructure/third_party/orchestration ./infrastructure/third_party/orchestration
COPY pkg ./pkg

# Copy extract service go mod files
COPY services/extract/go.mod services/extract/go.sum ./services/extract/

# Set working directory to extract service
WORKDIR /workspace/services/extract

# Download dependencies (skip if root go.mod missing)
RUN go mod download || go mod download -x || true

# Copy extract service source code
COPY services/extract/ ./

# Update go.mod and build the extract service
RUN go mod tidy && CGO_ENABLED=0 GOOS=linux go build -o server ./cmd/extract

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary
COPY --from=builder /build/server ./server

# Copy scripts
COPY services/extract/scripts ./scripts

# Make scripts executable
RUN chmod +x ./scripts/*.sh 2>/dev/null || true

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8082 9090 8815

# Entrypoint
ENTRYPOINT ["./scripts/start_extract.sh"]
