FROM golang:1.24-alpine AS builder

WORKDIR /workspace

RUN apk add --no-cache git build-base

COPY services/localai/go.mod services/localai/go.sum ./
RUN go mod download

COPY services/localai/ ./services/localai/

WORKDIR /workspace/services/localai

RUN CGO_ENABLED=0 go build -o /workspace/config-sync ./cmd/config-sync

FROM alpine:3.22

RUN apk add --no-cache ca-certificates

COPY --from=builder /workspace/config-sync /usr/local/bin/config-sync

ENTRYPOINT ["/usr/local/bin/config-sync"]

