FROM golang:1.24-alpine AS builder

WORKDIR /workspace

# Install goose for migrations
RUN go install github.com/pressly/goose/v3/cmd/goose@latest

# Root module files (for replace directives)
COPY go.mod go.sum go.work ./

# Local module dependencies referenced via replace directives
COPY pkg ./pkg
COPY services/extract ./services/extract
COPY services/orchestration ./services/orchestration
COPY infrastructure/third_party/go-hdb ./infrastructure/third_party/go-hdb

# Catalog module files
COPY services/catalog/go.mod services/catalog/go.sum ./services/catalog/
COPY services/catalog ./services/catalog

WORKDIR /workspace/services/catalog

RUN go mod tidy

# Build
RUN GOWORK=off go build -o catalog-service .

FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy catalog service binary
COPY --from=builder /app/catalog-service .

# Copy goose binary (for migrations)
COPY --from=builder /go/bin/goose /usr/local/bin/goose

# Copy migrations directory (from services/catalog/migrations)
COPY services/catalog/migrations ./migrations

EXPOSE 8084

ENTRYPOINT ["./catalog-service"]

