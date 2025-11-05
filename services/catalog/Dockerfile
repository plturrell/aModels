FROM golang:1.22-alpine AS builder

WORKDIR /app

# Install goose for migrations
RUN go install github.com/pressly/goose/v3/cmd/goose@latest

# Copy go.mod and go.sum
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build
RUN GOWORK=off go build -o catalog-service .

FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy catalog service binary
COPY --from=builder /app/catalog-service .

# Copy goose binary (for migrations)
COPY --from=builder /go/bin/goose /usr/local/bin/goose

# Copy migrations directory
COPY migrations ./migrations

EXPOSE 8084

ENTRYPOINT ["./catalog-service"]

