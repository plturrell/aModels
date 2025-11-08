# Maths Infrastructure Makefile
# Provides verification, testing, and benchmarking targets

.PHONY: verify fmt vet test bench coverage clean

# Default target
verify: fmt vet test

# Code formatting
fmt:
	@echo "Formatting maths package..."
	go fmt ./...

# Static analysis
vet:
	@echo "Running go vet on maths package..."
	go vet ./...

# Run all tests
test:
	@echo "Running maths package tests..."
	go test ./...

# Run tests with coverage
coverage:
	@echo "Running tests with coverage..."
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Run benchmarks
bench:
	@echo "Running maths benchmarks..."
	go test -bench=. -benchmem ./...

# Run specific SIMD benchmarks
bench-simd:
	@echo "Running SIMD benchmarks..."
	go test -bench='SIMD|Fused|Vectorized' -benchmem ./...

# Run matrix multiplication benchmarks
bench-matrix:
	@echo "Running matrix multiplication benchmarks..."
	go test -bench='MatMul|GEMM|Dot' -benchmem ./...

# Run attention benchmarks
bench-attention:
	@echo "Running attention benchmarks..."
	go test -bench='Attention|FlashAttention|Softmax' -benchmem ./...

# Compare against baseline (requires baseline results)
bench-compare:
	@echo "Comparing against baseline..."
	@if [ ! -f baseline.txt ]; then \
		echo "No baseline.txt found. Run 'make bench-baseline' first."; \
		exit 1; \
	fi
	go test -bench=. -benchmem ./... | tee current.txt
	@echo "Comparing current vs baseline..."
	@if command -v benchstat >/dev/null 2>&1; then \
		benchstat baseline.txt current.txt; \
	else \
		echo "Install benchstat: go install golang.org/x/perf/cmd/benchstat@latest"; \
	fi

# Generate baseline for comparison
bench-baseline:
	@echo "Generating baseline benchmarks..."
	go test -bench=. -benchmem -count=5 ./... | tee baseline.txt
	@echo "Baseline saved to baseline.txt"

# Run integration tests
integration:
	@echo "Running integration tests..."
	go test -tags integration ./...

# Run all benchmarks with different SIMD configurations
bench-all-configs:
	@echo "Running benchmarks with different SIMD configurations..."
	@echo "Default (AVX2 on x86-64, NEON on ARM64):"
	go test -bench=. -benchmem ./...
	@echo "AVX-512 (if supported):"
	go test -tags simd_asm_avx512 -bench=. -benchmem ./...
	@echo "NEON assembly (ARM64):"
	go test -tags 'simd_asm_neon asm_neon' -bench=. -benchmem ./...

# Performance profiling
profile:
	@echo "Running performance profiling..."
	go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof ./...
	@echo "Profiles generated: cpu.prof, mem.prof"
	@echo "View with: go tool pprof cpu.prof"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f coverage.out coverage.html
	rm -f baseline.txt current.txt
	rm -f *.prof
	rm -f *.test

# Install benchmark tools
install-tools:
	@echo "Installing benchmark tools..."
	go install golang.org/x/perf/cmd/benchstat@latest
	go install github.com/google/pprof@latest

# Help
help:
	@echo "Available targets:"
	@echo "  verify        - Run fmt, vet, and test"
	@echo "  fmt           - Format code"
	@echo "  vet           - Run static analysis"
	@echo "  test          - Run all tests"
	@echo "  coverage      - Run tests with coverage report"
	@echo "  bench         - Run all benchmarks"
	@echo "  bench-simd    - Run SIMD-specific benchmarks"
	@echo "  bench-matrix  - Run matrix operation benchmarks"
	@echo "  bench-attention - Run attention mechanism benchmarks"
	@echo "  bench-compare - Compare against baseline"
	@echo "  bench-baseline - Generate baseline for comparison"
	@echo "  bench-all-configs - Run benchmarks with different SIMD configs"
	@echo "  integration   - Run integration tests"
	@echo "  profile       - Run performance profiling"
	@echo "  clean         - Clean generated files"
	@echo "  install-tools - Install benchmark tools"
	@echo "  help          - Show this help"
