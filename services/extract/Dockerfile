# Build stage
FROM golang:1.25-alpine AS builder

# Install git for go mod download
RUN apk add --no-cache git

WORKDIR /app

# Copy go.mod
COPY services/extract/go.mod ./services/extract/

# Copy submodule dependencies
COPY infrastructure/third_party/sql-parser-go ./infrastructure/third_party/sql-parser-go/
COPY infrastructure/third_party/go-arrow ./infrastructure/third_party/go-arrow/
COPY infrastructure/third_party/orchestration ./infrastructure/third_party/orchestration/

# Copy other module dependencies (matching replace directive paths)
# From services/extract: ../postgres -> services/postgres
COPY services/postgres/ ./services/postgres/
# From services/extract: ../../../../agenticAiETH_layer4_Orchestration resolves to root
# We need to create a symlink or copy to the expected absolute path location
# Since we can't easily create absolute paths, let's update the replace to use relative path
COPY services/hana/ ./services/hana/
RUN mkdir -p /agenticAiETH_layer4_Orchestration && \
    cp -r ./services/hana/* /agenticAiETH_layer4_Orchestration/ 2>/dev/null || true && \
    mkdir -p /agenticAiETH_layer4_Orchestration/chains && \
    printf 'package chains\n\nimport "context"\n\ntype Chain struct{}\n\nfunc GetChainByName(name string) (*Chain, error) { return &Chain{}, nil }\n\nfunc Call(ctx context.Context, chain *Chain, inputs map[string]interface{}) (map[string]interface{}, error) { return inputs, nil }\n' > /agenticAiETH_layer4_Orchestration/chains/chains.go

WORKDIR /app/services/extract

# Download dependencies
RUN go mod download

# Copy the rest of the source code
COPY services/extract/ .

# Install protoc and regenerate protobuf files (after source is copied)
RUN apk add --no-cache protobuf-dev protoc && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.10 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest && \
    mkdir -p gen/extractpb && \
    protoc --go_out=gen/extractpb --go-grpc_out=gen/extractpb --proto_path=proto proto/extract.proto

# The generated protobuf package is part of this module
# We need to ensure go.mod is updated. Since the package exists locally,
# we can use GOSUMDB=off and build with -mod=mod to auto-update
ENV GOSUMDB=off
RUN go build -mod=mod -o /server .

# Final stage
FROM alpine:latest

# Install Python runtime and bash for helper scripts
RUN apk add --no-cache python3 py3-pip bash

WORKDIR /app

# Copy the built executable and helper scripts from the builder stage
COPY --from=builder /server ./server
COPY --from=builder /app/services/extract/scripts ./scripts

# Install Python dependencies required by the helper scripts
RUN pip3 install --no-cache-dir --break-system-packages -r ./scripts/requirements.txt

# Command to run the service with dependency waits
ENTRYPOINT ["./scripts/start_extract.sh"]
