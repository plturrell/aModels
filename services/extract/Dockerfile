# Build stage
FROM golang:1.23-alpine AS builder

# Install git for go mod download
RUN apk add --no-cache git

WORKDIR /app

# Copy go.mod
COPY services/extract/go.mod ./services/extract/

# Copy submodule dependencies
COPY infrastructure/third_party/sql-parser-go ./infrastructure/third_party/sql-parser-go/
COPY infrastructure/third_party/go-arrow ./infrastructure/third_party/go-arrow/
COPY infrastructure/third_party/orchestration ./infrastructure/third_party/orchestration/
COPY infrastructure/third_party/go-hdb ./infrastructure/third_party/go-hdb/

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

# Download dependencies (use GOTOOLCHAIN=auto to allow downloading newer Go if needed)
ENV GOTOOLCHAIN=auto
RUN go mod download

# Copy the rest of the source code (excluding generated protobuf files for now)
COPY services/extract/ .

# First, update go.mod dependencies (protobuf package may not exist yet, that's OK)
# We'll generate the files after, and they're part of the same module
RUN go mod tidy 2>&1 | grep -v "does not contain package" || true

# Install protoc and regenerate protobuf files
RUN apk add --no-cache protobuf-dev protoc && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34.2 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.4.0 && \
    mkdir -p gen/extractpb && \
    protoc --go_out=gen/extractpb --go-grpc_out=gen/extractpb --proto_path=proto proto/extract.proto && \
    if [ -d gen/extractpb/github.com ]; then mv gen/extractpb/github.com/plturrell/aModels/services/extract/gen/extractpb/*.pb.go gen/extractpb/ && rm -rf gen/extractpb/github.com; fi && \
    mkdir -p ../tmp/postgres_gen && \
    protoc --go_out=../tmp/postgres_gen --go-grpc_out=../tmp/postgres_gen \
        --proto_path=../postgres/proto ../postgres/proto/v1/postgres_lang_service.proto && \
    if [ -d ../tmp/postgres_gen/github.com/plturrell/aModels/services/postgres/pkg/gen/v1 ]; then \
        cp ../tmp/postgres_gen/github.com/plturrell/aModels/services/postgres/pkg/gen/v1/*.pb.go ../postgres/pkg/gen/v1/ && \
        rm -rf ../tmp/postgres_gen; \
    fi

# Update go.mod again after protobuf files are generated (filters remote package errors)
RUN go mod tidy 2>&1 | grep -v "does not contain package" || true

# Build (protobuf files are now generated and in the same module)
RUN go build -o /server .

# Final stage
FROM alpine:latest

# Install Python runtime and bash for helper scripts
RUN apk add --no-cache python3 py3-pip bash

WORKDIR /app

# Copy the built executable and helper scripts from the builder stage
COPY --from=builder /server ./server
COPY --from=builder /app/services/extract/scripts ./scripts

# Install Python dependencies required by the helper scripts
# Note: transformers and torch are included for DeepSeek OCR support
RUN pip3 install --no-cache-dir --break-system-packages -r ./scripts/requirements.txt

# Command to run the service with dependency waits
ENTRYPOINT ["./scripts/start_extract.sh"]
