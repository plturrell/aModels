# Postgres Lang Service

This directory contains the Postgres telemetry service copied from `agenticAiETH_layer4_Postgres`.

## Overview

The Postgres Lang Service provides:
- gRPC service for telemetry logging and querying
- FastAPI gateway for HTTP/REST access
- Database administration endpoints
- Operation analytics

Refer to the original [README.md](README.md) for detailed setup instructions.

## Quick Start

### gRPC Service

```bash
cd postgres
go run ./cmd/server/main.go
```

### FastAPI Gateway

```bash
cd postgres/gateway
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m gateway
```

## Integration

The unified browser gateway (`browser/gateway/`) includes an adapter for the Postgres data service, accessible at `/data/sql`. The gateway also aggregates telemetry at `/telemetry/recent`.

## Third-Party Libraries

### Arrow Flight

The postgres service uses **Apache Arrow v18.4.1** for high-performance data transfer via Arrow Flight.

**Usage**:
- Exposes operation logs via Arrow Flight server
- Endpoint: `operations/logs`
- Clients can use connection pooling (see `services/shared/pkg/pools/flight_pool.go`)

**Configuration**:
- Flight server address: Set via `FLIGHT_ADDR` environment variable
- Max rows per request: Configurable via `New()` function

**Client Usage**:
```go
// With connection pooling (recommended)
client, err := postgresflight.NewClient(addr, 10)
defer client.Close()
rows, err := client.FetchOperationsWithPool(ctx)

// Without pooling (backward compatible)
rows, err := postgresflight.FetchOperations(ctx, addr)
```

**See Also**: `docs/DEPENDENCY_MATRIX.md` for version compatibility information.
