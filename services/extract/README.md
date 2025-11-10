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
