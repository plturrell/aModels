FROM golang:1.24-alpine AS builder

WORKDIR /app

# Install goose for migrations
RUN go install github.com/pressly/goose/v3/cmd/goose@latest

# Copy go.mod and go.sum first
COPY services/catalog/go.mod services/catalog/go.sum ./

# Remove replace directives that point to missing local paths
RUN sed -i '/replace.*infrastructure\/third_party/d' go.mod || true && \
    sed -i '/replace.*third_party\/goose/d' go.mod || true && \
    sed -i '/replace github\.com\/plturrell\/aModels\/services\/catalog/d' go.mod || true

# Update go.mod and download dependencies
RUN go mod tidy && go mod download

# Copy source code
COPY services/catalog/ .

# Run go mod tidy to update go.mod after copying source
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

