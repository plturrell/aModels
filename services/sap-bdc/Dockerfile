FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY services/sap-bdc/go.mod services/sap-bdc/go.sum ./
RUN go mod download

# Copy source code
COPY services/sap-bdc/ .

# Build
RUN CGO_ENABLED=0 GOOS=linux go build -o sap-bdc-service ./main.go

FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

COPY --from=builder /app/sap-bdc-service .

EXPOSE 8083

CMD ["./sap-bdc-service"]

