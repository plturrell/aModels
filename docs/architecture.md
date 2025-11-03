# aModels System Architecture

## Overview

aModels is a microservices-based training and inference platform for AgenticAI Layer 4. The system is organized into clear architectural layers: services, data, infrastructure, tools, and testing.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Gateway  │  │AgentFlow │  │ Browser  │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────────┐
│                    Service Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Extract  │  │ Postgres │  │  HANA    │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│  ┌──────────┐                                          │
│  │ LocalAI  │                                          │
│  └──────────┘                                          │
└─────────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Training │  │ Eval Data│  │  Models   │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Docker  │  │ 3rd Party │  │  Cron    │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
```

## Module Organization

### 1. Services (`services/`)
All microservices grouped together for clear visibility:
- `agentflow/` - Workflow orchestration
- `extract/` - Data extraction service
- `gateway/` - Unified HTTP gateway
- `hana/` - SAP HANA integration
- `localai/` - AI inference server
- `postgres/` - PostgreSQL service
- `browser/` - Browser automation

### 2. Data (`data/`)
All data organized by purpose:
- `training/` - Training datasets (SGMI, etc.)
- `evaluation/` - Evaluation datasets (ARC-AGI, HellaSwag, etc.)
- `models/` - Model metadata and checkpoints

### 3. Infrastructure (`infrastructure/`)
Deployment and external dependencies:
- `docker/` - Docker Compose configurations
- `third_party/` - Git submodules (dependencies)
- `cron/` - Scheduled job definitions

### 4. Tools (`tools/`)
Development and utility tools:
- `scripts/` - Training and utility scripts
- `cmd/` - Command-line tools
- `helpers/` - Helper utilities

### 5. Testing (`testing/`)
All testing-related code:
- `benchmarks/` - Benchmark implementations
- `tests/` - Integration and unit tests

### 6. Documentation (`docs/`)
System documentation and guides

### 7. Legacy (`legacy/`)
Read-only legacy code:
- `stage3/` - Legacy search/graph services (read-only)

## Data Flow

```
Training Data (SGMI)
    ↓
Extract Service (Schema Replication)
    ↓
Postgres/HANA (Storage)
    ↓
AgentFlow (Orchestration)
    ↓
LocalAI (Inference)
    ↓
Browser (Automation)
```

## Service Communication

- **Gateway** → All services (HTTP proxy)
- **AgentFlow** → LocalAI (HTTP), Extract (HTTP)
- **Extract** → Postgres (gRPC), Redis (optional)
- **All services** → Gateway (health checks)

## Technology Stack

- **Languages**: Go, Python, TypeScript/React
- **Databases**: PostgreSQL, SAP HANA, Neo4j, Redis
- **AI/ML**: LocalAI, VaultGemma, Relational Transformer
- **Infrastructure**: Docker, Docker Compose
- **Dependencies**: Arrow, Elasticsearch, Glean, LangChain, LangFlow, LangGraph

## Deployment Architecture

### Development
```bash
docker compose -f docker/compose.yml up
```

### Production (GPU)
```bash
docker compose -f docker/compose.yml -f docker/compose.gpu.yml --profile gpu up
```

### Brev GPU Environment
```bash
docker compose -f docker/brev/docker-compose.yml up
```

## Key Design Principles

1. **Microservices**: Each service is independently deployable
2. **Gateway Pattern**: Unified entry point via gateway service
3. **Service Discovery**: Services register with gateway for routing
4. **Data Separation**: Training and evaluation data clearly separated
5. **Infrastructure as Code**: All deployment configs in `docker/`
6. **Documentation**: Each service has comprehensive README

## Development Workflow

1. **Service Development**: Work in `services/{service}/`
2. **Testing**: Add tests in `testing/tests/`
3. **Benchmarking**: Add benchmarks in `testing/benchmarks/`
4. **Documentation**: Update `docs/` and service READMEs
5. **Deployment**: Update `docker/compose.yml` for new services

## See Also

- [SERVICES.md](../SERVICES.md) - Detailed service registry
- [README.md](../README.md) - Quickstart guide
- Individual service READMEs in `services/{service}/README.md`

