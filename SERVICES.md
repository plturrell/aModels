# aModels Services

This repository contains the following microservices and components:

## Core Services

| Service | Port | Language | Purpose | Status |
|---------|------|----------|---------|--------|
| **gateway** | 8000 | Python | Unified HTTP gateway for all services | ✅ Active |
| **localai** | 8081 | Go | Local AI inference server (OpenAI-compatible) | ✅ Active |
| **extract** | 19080 | Go | Data extraction and schema replication | ✅ Active |
| **postgres** | 5432 | Go | PostgreSQL telemetry and gRPC service | ✅ Active |
| **hana** | 8070 | Go | SAP HANA database integration | ✅ Active |
| **agentflow** | 8001 | Go/Python | LangFlow workflow orchestration | ✅ Active |
| **browser** | 8070 | Go | Layer4 browser automation service | ✅ Active |
| **graph** | 8081 | Go | GPU-accelerated graph computation service | ✅ Active |
| **search** | 9200 | Java | Intelligent search and discovery platform | ✅ Active |

## Service Details

### Gateway (`gateway/`)
- **Purpose**: Unified HTTP gateway routing requests to all services
- **Endpoints**: `/healthz`, `/agentflow/*`, `/extract/*`, `/data/*`, `/search/*`, `/redis/*`, `/localai/*`, `/browser/*`
- **Tech**: FastAPI (Python)
- **Dependencies**: All other services

### LocalAI (`localai/`)
- **Purpose**: OpenAI-compatible local inference server
- **Endpoints**: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`
- **Tech**: Go (VaultGemma, quantized models)
- **Features**: GPU acceleration, streaming, embeddings

### Extract (`extract/`)
- **Purpose**: Data extraction, schema replication, OCR, SQL exploration
- **Endpoints**: `/ocr`, `/schema-replication`, `/sql`, `/graph`
- **Tech**: Go (gRPC + HTTP)
- **Features**: Glean integration, Neo4j, Redis caching

### Postgres (`postgres/`)
- **Purpose**: PostgreSQL telemetry and data service
- **Endpoints**: `/telemetry/recent`, gRPC service
- **Tech**: Go (gRPC + FastAPI gateway)
- **Features**: Telemetry collection, schema replication

### HANA (`hana/`)
- **Purpose**: SAP HANA database integration
- **Endpoints**: `/healthz`, `POST /sql`
- **Tech**: Go (go-hdb driver)
- **Features**: SQL query execution, health monitoring

### AgentFlow (`agentflow/`)
- **Purpose**: LangFlow workflow orchestration and process management
- **Endpoints**: `/run`, `/flows/*`
- **Tech**: Go (CLI) + Python (service) + React (frontend)
- **Features**: Flow execution, process catalog, telemetry

### Browser (`browser/`)
- **Purpose**: Layer4 browser automation and interaction
- **Endpoints**: `/healthz` (via gateway proxy)
- **Tech**: Go (Chromium automation)
- **Features**: Web automation, page interaction, content extraction

### Graph (`graph/`)
- **Purpose**: GPU-accelerated graph computation and LangGraph-Go port
- **Endpoints**: `/run`, `/agent/catalog`, `/extract/graph`, `/postgres/operations`
- **Tech**: Go (LangGraph runtime, Arrow Flight, HANA integration)
- **Features**: Graph execution, checkpointing, agent orchestration, Flight integration

### Search (`search/`)
- **Purpose**: Intelligent search and discovery platform with AI-powered search
- **Endpoints**: `/search`, `/qa`, `/embeddings`, `/inference`
- **Tech**: Java (Elasticsearch-based, modular architecture)
- **Features**: Hybrid search (keyword + semantic), QA, embeddings, inference, x-pack extensions

## Service Dependencies

```
gateway
  ├── agentflow
  ├── extract
  ├── postgres
  ├── localai
  ├── hana
  ├── browser
  ├── graph
  └── search

agentflow
  ├── localai
  └── extract

extract
  ├── postgres (optional)
  └── redis (optional)
```

## Deployment

All services can be deployed via Docker Compose:

```bash
# Base services
docker compose -f docker/compose.yml up

# With GPU support
docker compose -f docker/compose.yml -f docker/compose.gpu.yml --profile gpu up
```

## Health Checks

All services expose health endpoints:
- Gateway: `GET http://localhost:8000/healthz`
- LocalAI: `GET http://localhost:8081/healthz`
- Extract: `GET http://localhost:19080/healthz`
- Postgres: Via gateway `/data/telemetry/recent`
- HANA: `GET http://localhost:8070/healthz`
- AgentFlow: Via gateway `/agentflow/run`
- Browser: Via gateway `/browser/health`
- Graph: `GET http://localhost:8081/agent/catalog`
- Search: `GET http://localhost:9200/_search`

## Development

Each service has its own:
- `README.md` - Service-specific documentation
- `Dockerfile` - Container definition
- `Makefile` - Build and run commands
- `go.mod` / `requirements.txt` - Dependencies

