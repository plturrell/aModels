FROM golang:1.22-alpine AS builder

WORKDIR /app

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

COPY --from=builder /app/catalog-service .

EXPOSE 8084

ENTRYPOINT ["./catalog-service"]

