# Build Instructions

This document describes how to build the AI Training Framework and its gRPC/proto components.

## Prerequisites

### Required Tools

1. **Go** (1.25.3 or later)
   ```bash
   go version
   ```

2. **Protocol Buffers Compiler (protoc)**
   ```bash
   # macOS (Homebrew)
   brew install protobuf
   
   # Linux
   # Download from https://github.com/protocolbuffers/protobuf/releases
   
   # Verify installation
   protoc --version
   ```

3. **Go Protocol Buffer Plugins**
   ```bash
   # Install from the project
   make install-tools
   
   # Or install manually
   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
   
   # Ensure Go bin is in your PATH
   export PATH=$PATH:$(go env GOPATH)/bin
   ```

4. **Python** (3.10 or later) - For training scripts
   ```bash
   python3 --version
   pip install -r requirements.txt  # If requirements.txt exists
   ```

## Quick Start

### 1. Clone and Setup

```bash
cd /Users/user/Library/CloudStorage/Dropbox/agenticAiETH/agenticAiETH_layer4_Training
```

### 2. Install Dependencies

```bash
# Tidy Go dependencies
make tidy

# Install proto tools
make install-tools
```

### 3. Build Everything

```bash
# Build all packages
make build

# Or build all binaries
make build-all
```

## Build Targets

### Main Binaries

```bash
# Build aibench tool
make build-aibench

# Build benchmark server
make build-benchmark-server

# Build ARCagi service
make build-arcagi-service
```

### Protocol Buffers / gRPC

The project uses Protocol Buffers for:
- **Sentencepiece models**: Located in `models/sentencepiece`
- **Arrow Flight**: Used for Agent SDK catalog communication (not proto generation needed)

#### Generate Proto Bindings

```bash
# Generate all proto bindings
make proto

# Or generate proto + normalization data
make generate
```

This will:
1. Generate Go bindings for sentencepiece protos
2. Generate normalization data tables

#### Manual Proto Generation

```bash
cd models/sentencepiece
make proto
```

## Running Services

### AIBench Tool

```bash
# List available benchmarks
./aibench list

# Run a benchmark
./aibench run --task=arc --data=<path> --model=hybrid
```

### Benchmark Server

```bash
# Run with default settings
make run-benchmark-server

# Or run with custom settings
./benchmark-server \
  --port=8082 \
  --localai-url=http://localhost:8080 \
  --ui \
  --ui-port=8081
```

#### With Agent SDK Flight Integration

```bash
# Enable Agent SDK catalog
AGENTSDK_FLIGHT_ADDR=:53061 ./benchmark-server --agent-sdk-flight=:53061
```

### ARCagi Service

```bash
# Ensure HANA environment variables are set
# HDB_HOST, HDB_PORT, HDB_USER, HDB_PASSWORD
make run-arcagi-service
```

## Testing

```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage
# Opens coverage.html in browser
```

## Code Quality

```bash
# Format code
make fmt

# Vet code
make vet

# Run linter (requires golangci-lint)
make lint

# Run all checks
make check
```

## Project Structure

```
.
├── cmd/                      # Binary entry points
│   ├── aibench/             # Main benchmark tool
│   ├── benchmark-server/    # HTTP/WebSocket server
│   └── arcagi_service/      # HANA database service
├── internal/                # Internal packages
│   ├── catalog/            # Flight catalog integration
│   │   └── flightcatalog/  # Arrow Flight client
│   ├── localai/            # LocalAI integration
│   └── models/             # Model implementations
├── models/                  # Standalone model packages
│   ├── sentencepiece/      # Sentencepiece tokenizer (with proto)
│   ├── glove/              # GloVe embeddings
│   └── relational_transformer/ # Relational transformer
├── benchmarks/             # Benchmark implementations
├── data/                   # Benchmark datasets
├── scripts/                # Training and utility scripts
└── pkg/                    # Public packages
```

## gRPC and Arrow Flight

This project uses two RPC mechanisms:

### 1. Protocol Buffers (Sentencepiece)
- **Location**: `models/sentencepiece/internal/proto`
- **Purpose**: Serialize/deserialize sentencepiece models
- **Generation**: `make proto`

### 2. Arrow Flight (Agent SDK)
- **Purpose**: Catalog and tool discovery from Agent SDK
- **No code generation needed**: Uses Arrow Flight Go client
- **Connection**: Via `AGENTSDK_FLIGHT_ADDR` environment variable
- **Integration**: In `internal/catalog/flightcatalog`

## Troubleshooting

### Proto generation fails

```bash
# Check protoc installation
protoc --version

# Check protoc-gen-go installation
which protoc-gen-go
protoc-gen-go --version

# Reinstall tools
make install-tools
```

### Build fails with missing dependencies

```bash
# Update dependencies
make tidy

# Verify module
make verify
```

### Arrow Flight connection errors

```bash
# Check Agent SDK is running
# Verify AGENTSDK_FLIGHT_ADDR is correct
# Default: :53061
```

## Environment Variables

### Required for ARCagi Service
```bash
export HDB_HOST=your-hana-host
export HDB_PORT=30015
export HDB_USER=your-user
export HDB_PASSWORD=your-password
```

### Optional for Benchmark Server
```bash
export AGENTSDK_FLIGHT_ADDR=:53061  # Agent SDK Flight address
```

## Clean Up

```bash
# Remove build artifacts
make clean

# Remove all generated files including proto
cd models/sentencepiece && make clean
```

## CI/CD Integration

```bash
# Full build and test pipeline
make tidy
make generate
make build-all
make test
make check
```

## Additional Resources

- [README.md](./README.md) - Project overview
- [models/sentencepiece/README.md](./models/sentencepiece/README.md) - Sentencepiece details
- [docs/RELATIONAL_TRANSFORMER.md](./docs/RELATIONAL_TRANSFORMER.md) - Transformer training
- [Protocol Buffers](https://protobuf.dev/) - Proto documentation
- [Arrow Flight](https://arrow.apache.org/docs/format/Flight.html) - Flight RPC protocol

## Support

For issues or questions:
1. Check existing documentation in `docs/`
2. Review test files for usage examples
3. Check the codebase comments and godoc
