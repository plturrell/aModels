# Postgres Lang Service

[![Go Version](https://img.shields.io/badge/Go-1.23-blue.svg)](https://golang.org)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready telemetry service for tracking LangChain, LangGraph, LlamaIndex, and other AI framework operations.

## Overview

The Postgres Lang Service provides:

- **gRPC API** for high-performance telemetry logging
- **FastAPI Gateway** for HTTP/REST access from browsers and web clients
- **Apache Arrow Flight** for bulk data transfer
- **Database Administration** endpoints for schema inspection and queries
- **Real-time Analytics** with aggregations and performance trends
- **Structured Logging** with zerolog
- **Docker Support** for containerized deployment
- **Health Checks** and graceful shutdown

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Start all services (PostgreSQL, gRPC, FastAPI)
docker-compose up -d

# Verify health
curl http://localhost:8000/health

# View logs
docker-compose logs -f
```

Services available at:
- **gRPC**: `localhost:50055`
- **REST API**: `http://localhost:8000`
- **PostgreSQL**: `localhost:5432`

### Option 2: Local Development

#### 1. Start PostgreSQL

```bash
make postgres-start
```

#### 2. Run gRPC Service

```bash
make run
# Or manually:
POSTGRES_DSN="postgres://user@localhost:5432/postgres?sslmode=disable" \
GRPC_PORT=50055 \
go run ./cmd/server/main.go
```

#### 3. Run FastAPI Gateway

```bash
cd gateway
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
POSTGRES_LANG_SERVICE_ADDR="localhost:50055" python -m gateway
```

## Features

### Core Capabilities

- **Operation Logging**: Track AI framework operations with input/output, latency, errors
- **Session Tracking**: Group operations by session and workflow
- **Status Management**: Monitor running, success, and error states
- **Analytics**: Success rates, latency trends, error breakdowns by library type
- **Pagination**: Efficient cursor-based pagination for large datasets
- **Time-based Filtering**: Query operations within specific time windows
- **Cleanup**: Automated data retention management

### API Endpoints

#### gRPC Service (port 50055)

- `HealthCheck`: Service health and version
- `LogLangOperation`: Record a new operation
- `GetLangOperation`: Retrieve operation by ID
- `ListLangOperations`: Paginated operation listing with filters
- `GetAnalytics`: Aggregated statistics and trends
- `CleanupOperations`: Delete old operations

#### REST Gateway (port 8000)

- `GET /health`: Health check
- `GET /operations`: List operations (filterable)
- `GET /operations/{id}`: Get specific operation
- `POST /operations`: Log new operation
- `GET /analytics`: Get analytics summary
- `POST /cleanup`: Cleanup old data
- `GET /statuses`: List available status values
- `GET /db/*`: Database admin endpoints (optional)

See the [API Documentation](gateway/README.md) for detailed endpoint specifications.

## Testing

### Unit Tests

```bash
# Go tests
make test
# Or:
go test ./...

# Python tests
cd gateway
pytest
```

### Integration Tests

```bash
# Requires running PostgreSQL
docker-compose up -d postgres
go test -tags=integration ./...
```

### Test Coverage

```bash
# Go coverage
go test -cover ./...

# Python coverage
cd gateway
pytest --cov=gateway --cov-report=html
```

## Architecture

```
┌─────────────────┐
│  Web Clients    │
│  (Browser/curl) │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  FastAPI        │
│  Gateway        │◄────── Database Admin (optional)
│  Port 8000      │
└────────┬────────┘
         │ gRPC
         ▼
┌─────────────────┐      ┌──────────────────┐
│  gRPC Service   │      │  Arrow Flight    │
│  Port 50055     │      │  Port 8825       │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────┐
         │   PostgreSQL   │
         │   Port 5432    │
         └────────────────┘
```

### Components

1. **gRPC Service** (`cmd/server/main.go`)
   - High-performance gRPC server
   - Structured logging with zerolog
   - Graceful shutdown
   - Health checks
   - Configuration validation

2. **FastAPI Gateway** (`gateway/`)
   - REST/JSON interface
   - CORS support
   - gRPC client wrapper
   - Optional database admin

3. **Repository Layer** (`pkg/repository/`)
   - Database abstraction
   - Transaction management
   - Query optimization

4. **Service Layer** (`pkg/service/`)
   - Business logic
   - Proto ↔ Model conversion
   - Input validation

5. **Arrow Flight Server** (`pkg/flight/`)
   - Bulk data transfer
   - High-performance streaming
   - Connection pooling

## Configuration

### Environment Variables

#### gRPC Service

```bash
# Required
export POSTGRES_DSN="postgres://user:pass@host:5432/db?sslmode=disable"

# Optional
export GRPC_PORT=50055                           # Default: 50055
export FLIGHT_ADDR=":8825"                        # Default: :8825
export FLIGHT_MAX_ROWS=200                        # Default: 200
export SERVICE_VERSION="1.0.0"                    # Default: 0.1.0
export POSTGRES_MAX_OPEN_CONN=20                  # Default: 20
export POSTGRES_MAX_IDLE_CONN=10                  # Default: 10
export POSTGRES_CONN_MAX_LIFETIME_MINUTES=30      # Default: 30
export SHUTDOWN_GRACE_PERIOD_SECONDS=15           # Default: 15
```

#### FastAPI Gateway

```bash
# Required
export POSTGRES_LANG_SERVICE_ADDR="localhost:50055"

# Optional
export POSTGRES_LANG_DB_DSN="..."                 # For admin endpoints
export POSTGRES_DB_ALLOW_MUTATIONS=false          # Default: false
export POSTGRES_DB_DEFAULT_LIMIT=100              # Default: 100
export POSTGRES_LANG_GATEWAY_CORS="*"             # Default: *
export FASTAPI_HOST="0.0.0.0"                     # Default: 0.0.0.0
export FASTAPI_PORT=8000                          # Default: 8000
```

## Deployment

### Docker Compose

See [DOCKER_README.md](DOCKER_README.md) for comprehensive Docker deployment guide.

```bash
docker-compose up -d
```

### Kubernetes

Example manifests in [DOCKER_README.md](DOCKER_README.md#kubernetes-deployment).

### Systemd

```bash
# Install systemd units
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start services
sudo systemctl start agentic-postgres
sudo systemctl start agentic-layer4
sudo systemctl start agentic-gateway

# Enable on boot
sudo systemctl enable agentic-postgres agentic-layer4 agentic-gateway
```

## Security

⚠️ **Important**: This service does not include authentication by default.

See [SECURITY.md](SECURITY.md) for:
- Authentication implementation examples
- Database admin endpoint risks
- TLS/SSL configuration
- Secrets management
- Security checklist

### Quick Security Wins

```bash
# 1. Disable database admin endpoints in production
unset POSTGRES_LANG_DB_DSN

# 2. Use SSL for database
POSTGRES_DSN="postgres://user:pass@host/db?sslmode=require"

# 3. Restrict CORS
POSTGRES_LANG_GATEWAY_CORS="https://yourdomain.com"

# 4. Add authentication (see SECURITY.md)
```

## Monitoring

### Health Checks

```bash
# gRPC
./healthcheck -addr=localhost:50055 -timeout=2s

# REST
curl http://localhost:8000/health
```

### Logs

```bash
# Docker
docker-compose logs -f postgres-service

# Systemd
journalctl -u agentic-layer4 -f
```

### Metrics (Future)

- Prometheus integration planned
- Custom metrics for operation counts, latencies, errors

## Third-Party Libraries

### Apache Arrow Flight

High-performance data transfer using **Apache Arrow v18.4.1**.

**Usage**:
```go
client, err := postgresflight.NewClient(addr, 10)
defer client.Close()
rows, err := client.FetchOperationsWithPool(ctx)
```

**Configuration**: Set `FLIGHT_ADDR` and `FLIGHT_MAX_ROWS`.

### Dependencies

- **Go**: zerolog, pgx/v5, grpc, protobuf, arrow-go
- **Python**: fastapi, grpcio, psycopg2, uvicorn

See `go.mod` and `gateway/requirements.txt` for versions.
