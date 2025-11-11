# Extract Service

## Liquid Neural Network (LNN) for Terminology Learning

The Extract service includes a hierarchical Liquid Neural Network (LNN) for learning and inferring terminology patterns. The LNN consists of multiple layers:

- **Universal Layer**: Learns patterns common across all domains
- **Domain-Specific Layers**: Learn domain-specific terminology (e.g., "sales", "finance")
- **Naming Convention Layer**: Learns naming patterns (e.g., snake_case, camelCase)
- **Business Role Layer**: Learns business role classifications

### Features

- **Adam Optimizer**: Advanced optimization algorithm with momentum and adaptive learning rates
- **Batch Learning**: Gradient accumulation for more stable training
- **Attention Mechanisms**: Scaled dot-product attention for better context understanding
- **Model Persistence**: Save and load trained models
- **Memory Efficiency**: Sparse vocabulary with automatic pruning
- **Reproducibility**: Seeded random number generation for consistent results

### Configuration

Environment variables for LNN configuration:

- `LNN_RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `LNN_BATCH_SIZE`: Batch size for gradient accumulation (default: 32)
- `LNN_ADAM_LR`: Adam learning rate (default: 0.001)
- `LNN_ADAM_BETA1`: Adam beta1 parameter (default: 0.9)
- `LNN_ADAM_BETA2`: Adam beta2 parameter (default: 0.999)
- `LNN_USE_ATTENTION`: Enable attention mechanisms (default: true, set to "false" to disable)
- `LNN_MAX_VOCAB_SIZE`: Maximum vocabulary size before pruning (default: 10000)

### Usage Example

```go
// Create LNN instance
logger := log.New(os.Stdout, "", log.LstdFlags)
tnn := NewTerminologyLNN(logger)

// Learn domain-specific terminology
ctx := context.Background()
err := tnn.LearnDomain(ctx, "customer_id", "customers", "sales", time.Now())
if err != nil {
    log.Fatal(err)
}

// Infer domain from column name
domain, confidence := tnn.InferDomain(ctx, "customer_id", "customers", nil)
fmt.Printf("Domain: %s (confidence: %.2f)\n", domain, confidence)

// Save model
err = tnn.SaveModel("/path/to/model.json")
if err != nil {
    log.Fatal(err)
}

// Load model
tnn2 := NewTerminologyLNN(logger)
err = tnn2.LoadModel("/path/to/model.json")
if err != nil {
    log.Fatal(err)
}
```

### Model Persistence

Models are saved in JSON format with versioning support. The saved model includes:
- Layer weights and states
- Optimizer state (momentum, velocity, step count)
- Attention layer weights (if enabled)
- Vocabulary mappings

### Performance Tuning

1. **Batch Size**: Increase `LNN_BATCH_SIZE` for more stable gradients but slower updates
2. **Learning Rate**: Adjust `LNN_ADAM_LR` based on convergence speed (lower = slower but more stable)
3. **Attention**: Disable attention (`LNN_USE_ATTENTION=false`) for faster inference if context is not critical
4. **Vocabulary Size**: Adjust `LNN_MAX_VOCAB_SIZE` based on available memory

### Testing

Run the test suite:
```bash
go test -v ./terminology_lnn_test.go
```

Run benchmarks:
```bash
go test -bench=. -benchmem ./terminology_lnn_test.go
```

# Extract Service

This directory contains the Extract service copied from `agenticAiETH_layer4_Extract`.

## Overview

The Extract service provides:
- OCR (Optical Character Recognition) capabilities
- Schema replication from databases
- SQL exploration and normalization
- SGMI view lineage generation
- Document embedding and persistence

Refer to the original [README.md](README.md) for detailed setup instructions.

## Quick Start

```bash
cd extract
go run main.go
```

Or via Docker:

```bash
docker build -t extract-service .
docker run -p 8081:8081 extract-service
```

## Integration

The unified browser gateway (`browser/gateway/`) includes adapters for Extract:
- `/extract/ocr` - OCR extraction
- `/extract/schema-replication` - Schema replication

The service also exposes a gRPC interface on port 9090 (default) and an HTTP/JSON interface on port 8081.

## Third-Party Libraries

### Arrow Flight

The extract service uses **Apache Arrow v18.4.1** for high-performance data transfer via Arrow Flight.

**Usage**:
- Exposes graph nodes and edges via Arrow Flight server
- Endpoints: `graph/nodes` and `graph/edges`
- Uses connection pooling for client connections (see `services/shared/pkg/pools/flight_pool.go`)

**Configuration**:
- Flight server address: Set via `FLIGHT_ADDR` environment variable
- Connection pooling: Configured via `services/graph/pkg/clients/extractflight/client.go`

**Optimizations**:
- Streaming for large datasets (batches of 1000 records)
- Zero-copy where possible
- Connection reuse via pooling

**See Also**: `docs/DEPENDENCY_MATRIX.md` for version compatibility information.

### Goose (Database Migrations)

The extract service uses **Goose v3.21.1** for database migrations.

**Usage**:
- Migration files: `migrations/`
- Run migrations: `goose -dir migrations up`

**Configuration**:
- Database connection: Set via environment variables
- Migration directory: `migrations/`

**See Also**: `services/catalog` for similar Goose usage patterns.
