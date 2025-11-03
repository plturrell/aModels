.PHONY: all build build-all test clean proto help install-tools tidy verify
.PHONY: build-aibench build-benchmark-server build-arcagi-service
.PHONY: run-aibench run-benchmark-server run-arcagi-service

# Default target
all: build

# Build all packages
build:
	@echo "Building all packages..."
	go build ./...

# Build all binaries
build-all: build-aibench build-benchmark-server build-arcagi-service
	@echo "All binaries built successfully"

# Build individual binaries
build-aibench:
	@echo "Building aibench..."
	go build -o aibench ./cmd/aibench

build-benchmark-server:
	@echo "Building benchmark-server..."
	go build -o benchmark-server ./cmd/benchmark-server

build-arcagi-service:
	@echo "Building arcagi_service..."
	go build -o arcagi_service ./cmd/arcagi_service

# Run binaries
run-aibench:
	go run ./cmd/aibench

run-benchmark-server:
	go run ./cmd/benchmark-server

run-arcagi-service:
	go run ./cmd/arcagi_service

# Run tests
test:
	@echo "Running tests..."
	go test ./...

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f aibench benchmark-server arcagi_service
	rm -f coverage.out coverage.html
	go clean ./...

# Generate protobuf bindings for sentencepiece
# Note: Proto files are already generated in models/sentencepiece/internal/proto
# This target requires source proto files from upstream sentencepiece
proto:
	@echo "Checking protobuf bindings..."
	@if [ -f models/sentencepiece/internal/proto/sentencepiece.pb.go ]; then \
		echo "Proto files already generated (sentencepiece.pb.go exists)"; \
		echo "Skipping regeneration - proto files are up to date"; \
	elif [ -d models/sentencepiece ]; then \
		echo "Generating protobuf bindings..."; \
		cd models/sentencepiece && \
		export PATH=$$PATH:$(shell go env GOPATH)/bin && \
		$(MAKE) proto || echo "Proto generation failed - source proto files may be missing"; \
	else \
		echo "Sentencepiece module not found, skipping proto generation"; \
	fi

# Generate all code (proto + normalization data)
generate: proto
	@echo "Generating normalization data..."
	@if [ -d models/sentencepiece ]; then \
		cd models/sentencepiece && $(MAKE) normdata; \
	fi
	@echo "Code generation complete"

# Tidy dependencies
tidy:
	@echo "Tidying dependencies..."
	go mod tidy

# Verify module
verify:
	@echo "Verifying module..."
	go mod verify

# Install required tools for proto generation
install-tools:
	@echo "Installing protoc tools..."
	@echo "Note: protoc must be installed separately from https://grpc.io/docs/protoc-installation/"
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	@echo "Tools installed to: $$(go env GOPATH)/bin"
	@echo "Make sure $$(go env GOPATH)/bin is in your PATH"

# Run linter (requires golangci-lint)
lint:
	@which golangci-lint > /dev/null || (echo "golangci-lint not installed. Install from https://golangci-lint.run/usage/install/" && exit 1)
	golangci-lint run ./...

# Format code
fmt:
	@echo "Formatting code..."
	go fmt ./...

# Vet code
vet:
	@echo "Vetting code..."
	go vet ./...

# Check code quality
check: fmt vet lint test
	@echo "All checks passed"

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build all packages (default)"
	@echo "  build            - Build all packages"
	@echo "  build-all        - Build all binaries"
	@echo "  build-aibench    - Build aibench binary"
	@echo "  build-benchmark-server - Build benchmark-server binary"
	@echo "  build-arcagi-service - Build arcagi_service binary"
	@echo "  run-aibench      - Run aibench"
	@echo "  run-benchmark-server - Run benchmark-server"
	@echo "  run-arcagi-service - Run arcagi_service"
	@echo "  test             - Run tests"
	@echo "  test-coverage    - Run tests with coverage report"
	@echo "  clean            - Clean build artifacts"
	@echo "  proto            - Generate protobuf bindings"
	@echo "  generate         - Generate all code (proto + normdata)"
	@echo "  install-tools    - Install protoc tools"
	@echo "  tidy             - Tidy dependencies"
	@echo "  verify           - Verify module"
	@echo "  lint             - Run linter"
	@echo "  fmt              - Format code"
	@echo "  vet              - Vet code"
	@echo "  check            - Run all checks (fmt, vet, lint, test)"
	@echo "  help             - Show this help message"
