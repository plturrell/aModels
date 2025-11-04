# DeepAgents Service

Deep agent service for aModels with planning, sub-agent spawning, and integration with Knowledge Graphs, AgentFlow, and Orchestration.

## Overview

This service provides a Python-based deep agent built on [deepagents](https://pypi.org/project/deepagents/) that integrates with the aModels platform. It enables:

- **Planning & Task Decomposition**: Break down complex tasks using todo lists
- **Sub-Agent Spawning**: Create specialized sub-agents for focused tasks
- **File System Access**: Manage context with file system tools
- **Knowledge Graph Integration**: Query Neo4j knowledge graphs
- **AgentFlow Integration**: Run LangFlow flows
- **Orchestration Integration**: Execute orchestration chains

## Features

### Built-in Tools

1. **Knowledge Graph Tool** (`query_knowledge_graph`)
   - Query Neo4j using Cypher
   - Find tables, columns, relationships
   - Analyze data lineage and quality

2. **AgentFlow Tool** (`run_agentflow_flow`)
   - Execute pre-configured LangFlow flows
   - Run data pipeline workflows

3. **Orchestration Tool** (`run_orchestration_chain`)
   - Execute orchestration chains
   - Question answering, summarization, analysis

### DeepAgents Built-in Tools

- `write_todos`: Plan and track complex tasks
- `task`: Spawn specialized sub-agents
- `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`: File system operations

## Environment Variables

```bash
# Service URLs
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081

# Model Configuration
ANTHROPIC_API_KEY=your_key  # Preferred
OPENAI_API_KEY=your_key     # Alternative
LOCALAI_URL=http://localai:8080  # Fallback

# Service Port
DEEPAGENTS_PORT=9004
```

## API Endpoints

### `GET /healthz`
Health check endpoint.

### `POST /invoke`
Invoke the deep agent with a conversation.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Analyze the SGMI pipeline"}
  ],
  "stream": false,
  "config": {}
}
```

**Response:**
```json
{
  "messages": [
    {"role": "assistant", "content": "..."}
  ],
  "result": {...}
}
```

### `POST /stream`
Stream agent responses (Server-Sent Events).

### `GET /agent/info`
Get information about the configured agent.

## Usage Examples

### Example 1: Pipeline Analysis

```python
import httpx

client = httpx.Client()

response = client.post(
    "http://localhost:9004/invoke",
    json={
        "messages": [
            {
                "role": "user",
                "content": """Analyze the SGMI pipeline:
                1. Query knowledge graph for all Control-M jobs
                2. Find associated SQL queries
                3. Analyze data flow
                4. Generate optimization recommendations"""
            }
        ]
    }
)

result = response.json()
print(result["messages"][-1]["content"])
```

### Example 2: Data Quality Assessment

```python
response = client.post(
    "http://localhost:9004/invoke",
    json={
        "messages": [
            {
                "role": "user",
                "content": """Assess data quality for the SGMI system:
                1. Query knowledge graph for all tables
                2. For each table, analyze data quality metrics
                3. Use orchestration chain to generate recommendations
                4. Create a summary report"""
            }
        ]
    }
)
```

### Example 3: Automated Flow Generation

```python
response = client.post(
    "http://localhost:9004/invoke",
    json={
        "messages": [
            {
                "role": "user",
                "content": """Create an AgentFlow flow for the Control-M pipeline:
                1. Query knowledge graph for Control-M jobs and SQL
                2. Analyze dependencies
                3. Generate LangFlow flow JSON
                4. Create flow in AgentFlow service"""
            }
        ]
    }
)
```

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn main:app --host 0.0.0.0 --port 9004 --reload
```

### Docker

```bash
# Build image
docker build -t amodels-deepagents .

# Run container
docker run -p 9004:9004 \
  -e EXTRACT_SERVICE_URL=http://extract-service:19080 \
  -e AGENTFLOW_SERVICE_URL=http://agentflow-service:9001 \
  -e GRAPH_SERVICE_URL=http://graph-service:8081 \
  -e ANTHROPIC_API_KEY=your_key \
  amodels-deepagents
```

## Integration with aModels

### Gateway Integration

The gateway provides `/deepagents/invoke` endpoint that proxies to this service.

### LangGraph Workflow Integration

Deep agents can be integrated into LangGraph workflows via the `DeepAgentNode`.

### Unified Workflow

Deep agents can be part of the unified workflow alongside Knowledge Graph processing, Orchestration, and AgentFlow.

## Architecture

```
┌─────────────────┐
│   Gateway       │
│   (FastAPI)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────────┐
│ Graph │ │ DeepAgents   │
│(Lang  │ │ Service      │
│Graph) │ │ (Python)     │
└───┬───┘ └──┬───────────┘
    │        │
    └────┬───┘
         │
┌────────▼────────────┐
│  Unified Workflow   │
│  (LangGraph)        │
└─────────────────────┘
```

## References

- [deepagents PyPI](https://pypi.org/project/deepagents/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [aModels Documentation](../README.md)

