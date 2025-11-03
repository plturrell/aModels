.PHONY: all build test clean proto normdata

# Default target
all: build

# Build all packages
build:
	go build ./...

# Run tests
test:
	go test ./...

# Run tests with coverage
coverage:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	rm -f coverage.out coverage.html
	go clean ./...

# Generate protobuf bindings
proto:
	@echo "Generating protobuf bindings..."
	@export PATH=$$PATH:$(shell go env GOPATH)/bin && go run ./cmd/gen_proto

# Generate normalization data
normdata:
	@echo "Generating normalization data..."
	@go run ./cmd/gen_normdata

# Generate all code
generate: proto normdata
	@echo "Code generation complete"

# Run linter
lint:
	golangci-lint run ./...

# Format code
fmt:
	go fmt ./...

# Tidy dependencies
tidy:
	go mod tidy

# Verify module
verify:
	go mod verify

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build all packages (default)"
	@echo "  build     - Build all packages"
	@echo "  test      - Run tests"
	@echo "  coverage  - Run tests with coverage report"
	@echo "  clean     - Clean build artifacts"
	@echo "  proto     - Generate protobuf bindings (TODO)"
	@echo "  normdata  - Generate normalization data (TODO)"
	@echo "  lint      - Run linter"
	@echo "  fmt       - Format code"
	@echo "  tidy      - Tidy dependencies"
	@echo "  verify    - Verify module"
	@echo "  help      - Show this help message"
