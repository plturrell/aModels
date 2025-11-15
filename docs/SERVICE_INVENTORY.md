# aModels Service Inventory

Generated: $(date)

## Overview

This document provides a comprehensive inventory of all services in the aModels system, including their configuration, ports, health endpoints, and startup mechanisms.

## Services in Registry (config/services.yaml)

### Infrastructure Services

| Service | Port(s) | Health Endpoint | Dependencies | Required | Notes |
|---------|---------|----------------|--------------|----------|-------|
| Redis | 6379 | tcp://localhost:6379 | None | Yes | Cache and message broker |
| PostgreSQL | 5432 | postgresql://postgres:postgres@localhost:5432/amodels | None | Yes | Primary database |
| Neo4j | 7474, 7687 | http://localhost:7474 | None | Yes | Graph database |
| Elasticsearch | 9200, 9300 | http://localhost:9200/_cluster/health | None | No | Search engine |
| Gitea | 3003, 2222 | http://localhost:3003/api/healthz | postgres | No | Git repository service |

### Core Services

| Service | Port(s) | Health Endpoint | Dependencies | Required | Notes |
|---------|---------|----------------|--------------|----------|-------|
| LocalAI | 8081 | http://localhost:8081/healthz | redis, postgres | Yes | AI model server |
| Transformers | 9090 | http://localhost:9090/health | None | No | GPU-required transformer service |
| Catalog | 8084 | http://localhost:8084/health | neo4j, redis, postgres | Yes | Metadata catalog service |

### Application Services

| Service | Port(s) | Health Endpoint | Dependencies | Required | Notes |
|---------|---------|----------------|--------------|----------|-------|
| Extract | 8083 | http://localhost:8083/health | elasticsearch, neo4j, postgres, catalog | Yes | Document extraction service |
| Graph | 8080, 19080 | http://localhost:8080/health | neo4j, localai | Yes | Graph operations service (GPU-required) |
| Search Inference | 8090 | http://localhost:8090/health | elasticsearch, localai | No | Search inference service |
| DeepAgents | 9004 | http://localhost:9004/healthz | postgres, redis, localai | No | Deep learning agents service |
| Runtime | 8098 | http://localhost:8098/healthz | catalog | No | Runtime analytics server |
| Orchestration | 8085 | http://localhost:8085/healthz | None | No | Workflow orchestration server |
| Training | 8087 | http://localhost:8087/health | postgres, redis, localai, extract | No | Model training service |
| PostgreSQL Lang | 50051 | grpc://localhost:50051 | postgres | No | PostgreSQL language service (gRPC) |
| Regulatory Audit | 8099 | http://localhost:8099/healthz | neo4j, localai | No | BCBS239 regulatory audit server |

### Gateway Services

| Service | Port(s) | Health Endpoint | Dependencies | Required | Notes |
|---------|---------|----------------|--------------|----------|-------|
| Gateway | 8000 | http://localhost:8000/healthz | catalog, localai | No | API gateway service |

## Services in Directory but NOT in Registry

The following services exist in `/home/aModels/services/` but are **NOT** registered in `config/services.yaml`:

### Missing from Registry

1. **telemetry-exporter** - Telemetry export service
   - Port: 8085 (default), 8080 (server mode), 8083 (docker-compose)
   - Health: `/health`
   - Type: Go service
   - Status: Needs to be added to registry
   - Note: Port ambiguity exists - default is 8085, but server mode uses 8080, docker-compose uses 8083

2. **agentflow** - Agent flow service
   - Type: Go service with frontend
   - Status: Needs investigation

3. **markitdown-service** - Markdown conversion service
   - Type: Python service
   - Status: Needs investigation

4. **gpu-orchestrator** - GPU orchestration service
   - Type: Go service
   - Status: Needs investigation

5. **hana** - HANA database service
   - Type: Go service
   - Status: Needs investigation

6. **sap-bdc** - SAP BDC service
   - Type: Unknown
   - Status: Needs investigation

7. **analytics** - Analytics service
   - Type: Go service
   - Status: Needs investigation

8. **plot** - Plot service
   - Type: Go service
   - Status: Needs investigation

9. **browser** - Browser automation service
   - Type: Multiple (Python, JavaScript)
   - Status: Complex service with multiple components

10. **testing** - Testing service
    - Type: Go service
    - Status: May be test infrastructure only

11. **shared** - Shared libraries
    - Type: Go/Python libraries
    - Status: Not a service, shared code

12. **framework** - Framework code
    - Type: Go libraries
    - Status: Not a service, framework code

13. **stdlib** - Standard library
    - Type: Libraries
    - Status: Not a service, library code

14. **third_party** - Third party code
    - Type: Various
    - Status: Not a service, third party code

## Health Check Script Status

### Services in health-check.sh

Currently checked:
- Redis
- PostgreSQL
- Neo4j (HTTP and Bolt)
- Elasticsearch
- LocalAI
- Catalog
- Extract
- Graph
- Search
- DeepAgents
- Runtime
- Orchestration
- Training
- DMS (deprecated)
- Regulatory Audit

### Services MISSING from health-check.sh

- Gitea
- Telemetry Exporter
- Gateway
- PostgreSQL Lang (gRPC)
- Transformers

## Port Conflicts and Issues

### Port Conflicts Identified

1. **Port 8080**: Used by both Graph service and Telemetry Exporter (server mode)
2. **Port 8083**: Used by Extract service and Telemetry Exporter (mentioned in docs)
3. **Port 8085**: Used by Orchestration service and Telemetry Exporter (default in code)

### Resolution Needed

- Telemetry Exporter port needs to be standardized
- Verify actual ports in use by checking running services

## Startup Profiles

### Minimal Profile
Services: redis, postgres, neo4j, localai, catalog

### Development Profile
Services: redis, postgres, neo4j, elasticsearch, localai, catalog, extract, search_inference, runtime, orchestration

### Full Profile
Services: redis, postgres, neo4j, elasticsearch, localai, transformers, catalog, extract, graph, search_inference, deepagents, runtime, orchestration, training, dms (deprecated), postgres_lang, regulatory_audit, gateway

## Next Steps

1. Add missing services to `config/services.yaml`:
   - telemetry-exporter
   - agentflow (if it's a service)
   - markitdown-service (if it's a service)
   - gpu-orchestrator (if it's a service)

2. Update `scripts/system/health-check.sh` to include:
   - Gitea
   - Telemetry Exporter
   - Gateway
   - PostgreSQL Lang (gRPC check)
   - Transformers

3. Resolve port conflicts:
   - Standardize Telemetry Exporter port
   - Document actual ports in use

4. Investigate unregistered services:
   - Determine if they are actual services or libraries
   - Add to registry if they are services
   - Document if they are libraries/frameworks

