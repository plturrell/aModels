# Configuration Reference

This document provides a comprehensive reference for all environment variables used across the lang infrastructure services.

## Overview

The lang infrastructure services use environment variables for configuration. This document lists all variables, their purposes, default values, and which services use them.

---

## Service URLs

### Core Service URLs

| Variable | Description | Default | Used By |
|----------|-------------|---------|---------|
| `EXTRACT_SERVICE_URL` | Extract service HTTP endpoint | `http://extract-service:19080` | DeepAgents, Graph, Extract |
| `AGENTFLOW_SERVICE_URL` | AgentFlow service HTTP endpoint | `http://agentflow-service:9001` | DeepAgents, Graph |
| `GRAPH_SERVICE_URL` | Graph service HTTP endpoint | `http://graph-service:8081` | DeepAgents |
| `LOCALAI_URL` | LocalAI service HTTP endpoint | `http://localai:8080` | All services |
| `DEEPAGENTS_URL` | DeepAgents service HTTP endpoint | `http://deepagents-service:9004` | Extract |
| `GPU_ORCHESTRATOR_URL` | GPU orchestrator service URL | - | Graph (optional) |

---

## DeepAgents Service

**Service Port**: 9004

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEEPAGENTS_PORT` | Service port | `9004` | No |
| `DEEPAGENTS_ENABLED` | Enable/disable service | `true` | No |
| `EXTRACT_SERVICE_URL` | Extract service URL | `http://extract-service:19080` | Yes |
| `AGENTFLOW_SERVICE_URL` | AgentFlow service URL | `http://agentflow-service:9001` | Yes |
| `GRAPH_SERVICE_URL` | Graph service URL | `http://graph-service:8081` | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key (preferred) | - | One of: Anthropic, OpenAI, or LocalAI |
| `OPENAI_API_KEY` | OpenAI API key (alternative) | - | One of: Anthropic, OpenAI, or LocalAI |
| `LOCALAI_URL` | LocalAI URL (fallback) | `http://localai:8080` | One of: Anthropic, OpenAI, or LocalAI |

**Example Configuration**:
```bash
DEEPAGENTS_PORT=9004
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081
ANTHROPIC_API_KEY=sk-ant-...
```

---

## AgentFlow Service

**Service Port**: 9001

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AGENTFLOW_PORT` | Service port | `9001` | No |
| `LANGFLOW_URL` | LangFlow backend URL (if external) | - | No |

**Example Configuration**:
```bash
AGENTFLOW_PORT=9001
LANGFLOW_URL=http://langflow:7860
```

---

## Orchestration Service

**Note**: This service is primarily used internally, not as a standalone HTTP service.

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GRAPH_SERVICE_URL` | Graph service URL | `http://graph-service:8081` | No |
| `DATABASE_URL` | Database connection (for digital twins) | - | No (if using digital twins) |
| `LOG_LEVEL` | Logging level | `info` | No |

**Example Configuration**:
```bash
GRAPH_SERVICE_URL=http://graph-service:8081
DATABASE_URL=postgres://user:pass@localhost/db
LOG_LEVEL=info
```

---

## Extract Service

**Service Ports**: 19080 (HTTP), 9090 (gRPC), 8815 (Arrow Flight)

### LangExtract Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LANGEXTRACT_URL` | LangExtract service URL | - | Yes |
| `LANGEXTRACT_API_KEY` | LangExtract API key | - | Yes |

### Knowledge Graph Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEO4J_URI` | Neo4j connection URI | - | Yes (if using Neo4j) |
| `NEO4J_USERNAME` | Neo4j username | - | Yes (if using Neo4j) |
| `NEO4J_PASSWORD` | Neo4j password | - | Yes (if using Neo4j) |

### Integration Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEEPAGENTS_URL` | DeepAgents service URL | `http://deepagents-service:9004` | No |
| `DEEPAGENTS_ENABLED` | Enable DeepAgents integration | `true` | No |
| `USE_SAP_RPT_EMBEDDINGS` | Enable semantic chain matching | `false` | No |
| `LOCALAI_URL` | LocalAI URL (for domain detection) | `http://localai:8080` | No |

### Persistence Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SQLITE_PATH` | SQLite database path | - | No |
| `REDIS_ADDR` | Redis address | - | No |
| `REDIS_PASSWORD` | Redis password | - | No |
| `REDIS_DB` | Redis database number | `0` | No |
| `DOC_STORE_PATH` | Document store path | - | No |

### Training Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `TRAINING_OUTPUT_DIR` | Training data output directory | - | No |

**Example Configuration**:
```bash
# LangExtract
LANGEXTRACT_URL=http://langextract-service:port
LANGEXTRACT_API_KEY=your_key

# Knowledge Graph
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Integration
DEEPAGENTS_URL=http://deepagents-service:9004
DEEPAGENTS_ENABLED=true
USE_SAP_RPT_EMBEDDINGS=true
LOCALAI_URL=http://localai:8080

# Persistence
SQLITE_PATH=/data/extract.db
REDIS_ADDR=redis:6379
DOC_STORE_PATH=/data/docs
```

---

## Graph Service

**Service Port**: 8081

### Service URLs

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `EXTRACT_SERVICE_URL` | Extract service URL | `http://extract-service:19080` | Yes |
| `AGENTFLOW_SERVICE_URL` | AgentFlow service URL | `http://agentflow-service:9001` | Yes |
| `LOCALAI_URL` | LocalAI service URL | `http://localai:8080` | Yes |
| `GPU_ORCHESTRATOR_URL` | GPU orchestrator URL | - | No |
| `DEEPAGENTS_SERVICE_URL` | DeepAgents service URL | `http://deepagents-service:9004` | No |

### Checkpoint Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `CHECKPOINT` | Checkpoint backend | `sqlite:langgraph.db` | No |
| `HANA_DSN` | HANA connection string (if using HANA) | - | No (if using HANA) |

**Checkpoint Options**:
- `sqlite:path/to/db` - SQLite (default for local dev)
- `redis://host:port/db` - Redis
- `hana` - SAP HANA (requires `-tags hana` build flag)

### External Service Integration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `EXTRACT_GRPC_ADDR` | Extract service gRPC endpoint | - | No |
| `EXTRACT_FLIGHT_ADDR` | Extract service Arrow Flight endpoint | - | No |
| `POSTGRES_GRPC_ADDR` | Postgres gRPC endpoint | - | No |
| `POSTGRES_FLIGHT_ADDR` | Postgres Arrow Flight endpoint | - | No |
| `AGENTSDK_FLIGHT_ADDR` | Agent SDK Arrow Flight endpoint | - | No |

**Example Configuration**:
```bash
# Service URLs
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
LOCALAI_URL=http://localai:8080
GPU_ORCHESTRATOR_URL=http://gpu-orchestrator:port

# Checkpoint
CHECKPOINT=sqlite:langgraph.db
# Or: CHECKPOINT=redis://localhost:6379/0
# Or: CHECKPOINT=hana (with HANA_DSN set)

# External Services (optional)
EXTRACT_GRPC_ADDR=extract-service:9090
EXTRACT_FLIGHT_ADDR=extract-service:8815
```

---

## LocalAI Service

**Service Port**: 8080

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LOCALAI_URL` | LocalAI service URL | `http://localai:8080` | Yes (for services using it) |

**Note**: LocalAI has its own internal configuration. This variable is used by other services to connect to LocalAI.

---

## Common Patterns

### Development Setup

```bash
# Core services
EXTRACT_SERVICE_URL=http://localhost:19080
AGENTFLOW_SERVICE_URL=http://localhost:9001
GRAPH_SERVICE_URL=http://localhost:8081
LOCALAI_URL=http://localhost:8080
DEEPAGENTS_URL=http://localhost:9004

# DeepAgents
ANTHROPIC_API_KEY=sk-ant-...

# Extract
LANGEXTRACT_URL=http://localhost:port
LANGEXTRACT_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### Docker Compose Setup

```bash
# Use service names as hostnames
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081
LOCALAI_URL=http://localai:8080
DEEPAGENTS_URL=http://deepagents-service:9004
```

### Production Setup

```bash
# Use load balancers or service mesh
EXTRACT_SERVICE_URL=https://extract.example.com
AGENTFLOW_SERVICE_URL=https://agentflow.example.com
GRAPH_SERVICE_URL=https://graph.example.com
LOCALAI_URL=https://localai.example.com
DEEPAGENTS_URL=https://deepagents.example.com

# Secure API keys
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}  # From secrets manager
LANGEXTRACT_API_KEY=${LANGEXTRACT_API_KEY}  # From secrets manager
```

---

## Validation

### Required Variables by Service

**DeepAgents**:
- `EXTRACT_SERVICE_URL`
- `AGENTFLOW_SERVICE_URL`
- `GRAPH_SERVICE_URL`
- At least one of: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `LOCALAI_URL`

**Extract**:
- `LANGEXTRACT_URL`
- `LANGEXTRACT_API_KEY`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` (if using Neo4j)

**Graph**:
- `EXTRACT_SERVICE_URL`
- `AGENTFLOW_SERVICE_URL`
- `LOCALAI_URL`

**AgentFlow**:
- No required variables (uses defaults)

**Orchestration**:
- No required variables (internal service)

---

## Environment Variable Naming Conventions

1. **Service URLs**: Use `{SERVICE}_SERVICE_URL` or `{SERVICE}_URL`
2. **API Keys**: Use `{SERVICE}_API_KEY`
3. **Ports**: Use `{SERVICE}_PORT`
4. **Feature Flags**: Use `{FEATURE}_ENABLED` (boolean)
5. **Paths**: Use `{COMPONENT}_PATH` or `{COMPONENT}_DIR`

---

## Configuration Files

### .env File Example

```bash
# Core Services
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081
LOCALAI_URL=http://localai:8080
DEEPAGENTS_URL=http://deepagents-service:9004

# DeepAgents
DEEPAGENTS_PORT=9004
ANTHROPIC_API_KEY=sk-ant-...

# Extract
LANGEXTRACT_URL=http://langextract-service:port
LANGEXTRACT_API_KEY=your_key
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
DEEPAGENTS_ENABLED=true
USE_SAP_RPT_EMBEDDINGS=true

# Graph
GPU_ORCHESTRATOR_URL=http://gpu-orchestrator:port
CHECKPOINT=sqlite:langgraph.db
```

---

## Troubleshooting

### Missing Required Variables

If a service fails to start, check:
1. All required variables are set
2. Variable names are correct (case-sensitive)
3. Values are properly formatted (no extra spaces, quotes if needed)

### Service Connection Issues

If services can't connect:
1. Verify service URLs are correct
2. Check network connectivity
3. Verify service ports match configuration
4. Review service logs for connection errors

### API Key Issues

If API calls fail:
1. Verify API keys are set correctly
2. Check API key permissions
3. Verify API quotas/limits
4. Review service logs for authentication errors

---

## References

- [DeepAgents Integration Guide](../services/deepagents/INTEGRATION.md)
- [AgentFlow Integration Guide](../services/agentflow/INTEGRATION.md)
- [Extract Integration Guide](../services/extract/INTEGRATION.md)
- [Graph Integration Guide](../services/graph/INTEGRATION.md)
- [Orchestration Integration Guide](../services/orchestration/INTEGRATION.md)

