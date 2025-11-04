# DeepAgents Integration

## Overview

DeepAgents is a Python-based deep agent service integrated into `aModels` that provides planning, sub-agent spawning, and file system capabilities. It complements the existing Go-based orchestration with advanced agent features.

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

## Features

### 1. Planning & Task Decomposition
- Built-in `write_todos` tool for breaking down complex tasks
- Track progress through multi-step workflows
- Adapt plans based on new information

### 2. Sub-Agent Spawning
- `task` tool for spawning specialized sub-agents
- Context isolation for focused tasks
- Examples: SQL optimization, data quality analysis, schema comparison

### 3. File System Access
- `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- Context management to prevent overflow
- Store intermediate results and final reports

### 4. Custom Tools Integration

#### Knowledge Graph Tool
- Query Neo4j using Cypher
- Find tables, columns, relationships
- Analyze data lineage and quality

#### AgentFlow Tool
- Run pre-configured LangFlow flows
- Execute data pipeline workflows

#### Orchestration Tool
- Execute orchestration chains
- Question answering, summarization, analysis

## API Endpoints

### Gateway Endpoints

#### `POST /deepagents/invoke`
Invoke the deep agent directly.

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

#### `POST /deepagents/stream`
Stream agent responses (Server-Sent Events).

#### `GET /deepagents/info`
Get information about the configured agent.

#### `POST /deepagents/process`
Process request via LangGraph workflow (includes knowledge graph context).

**Request:**
```json
{
  "deepagents_request": {
    "messages": [{"role": "user", "content": "..."}]
  },
  "knowledge_graph": {...}  // Optional: enrich with KG context
}
```

### Graph Service Endpoints

#### `POST /deepagents/process`
Process deep agent request via LangGraph workflow.

## Usage Examples

### Example 1: Pipeline Analysis

```bash
curl -X POST http://localhost:8000/deepagents/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Analyze the SGMI pipeline:\n1. Query knowledge graph for all Control-M jobs\n2. Find associated SQL queries\n3. Analyze data flow\n4. Generate optimization recommendations"
      }
    ]
  }'
```

### Example 2: Data Quality Assessment

```bash
curl -X POST http://localhost:8000/deepagents/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Assess data quality for the SGMI system:\n1. Query knowledge graph for all tables\n2. For each table, analyze data quality metrics\n3. Use orchestration chain to generate recommendations\n4. Create a summary report"
      }
    ]
  }'
```

### Example 3: Automated Flow Generation

```bash
curl -X POST http://localhost:8000/deepagents/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Create an AgentFlow flow for the Control-M pipeline:\n1. Query knowledge graph for Control-M jobs and SQL\n2. Analyze dependencies\n3. Generate LangFlow flow JSON\n4. Create flow in AgentFlow service"
      }
    ]
  }'
```

### Example 4: With Knowledge Graph Context

```bash
curl -X POST http://localhost:8000/deepagents/process \
  -H "Content-Type: application/json" \
  -d '{
    "deepagents_request": {
      "messages": [
        {
          "role": "user",
          "content": "Analyze this knowledge graph and suggest improvements"
        }
      ]
    },
    "knowledge_graph": {
      "nodes": [...],
      "edges": [...],
      "quality": {
        "score": 0.75,
        "level": "good"
      }
    }
  }'
```

## Integration with Unified Workflow

DeepAgents can be part of the unified workflow alongside Knowledge Graph processing, Orchestration, and AgentFlow:

```json
{
  "unified_request": {
    "knowledge_graph_request": {...},
    "orchestration_request": {...},
    "agentflow_request": {...},
    "deepagents_request": {
      "messages": [{"role": "user", "content": "..."}]
    },
    "workflow_mode": "parallel"
  }
}
```

## Configuration

### Environment Variables

```bash
# DeepAgents Service
DEEPAGENTS_URL=http://deepagents:9004
DEEPAGENTS_PORT=9004

# Service URLs (for DeepAgents tools)
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081

# Model Configuration
ANTHROPIC_API_KEY=your_key  # Preferred
OPENAI_API_KEY=your_key     # Alternative
LOCALAI_URL=http://localai:8080  # Fallback
```

## Docker Compose

The DeepAgents service is included in `infrastructure/docker/compose.yml`:

```yaml
deepagents:
  build:
    context: ../../services/deepagents
    dockerfile: Dockerfile
  environment:
    - EXTRACT_SERVICE_URL=http://extract-service:19080
    - AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
    - GRAPH_SERVICE_URL=http://graph-service:8081
    - LOCALAI_URL=http://localai:8080
    - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
    - OPENAI_API_KEY=${OPENAI_API_KEY:-}
  ports:
    - "9004:9004"
  depends_on:
    - postgres
    - redis
```

## Benefits

1. **Enhanced Planning**: Break down complex tasks into manageable steps
2. **Sub-Agent Isolation**: Specialized sub-agents for focused tasks
3. **Context Management**: File system tools prevent context overflow
4. **Long-term Memory**: Persistent memory across conversations
5. **Unified Interface**: Single entry point for complex agent tasks
6. **Python Ecosystem**: Access to Python LLM tools and libraries

## References

- [DeepAgents PyPI](https://pypi.org/project/deepagents/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [aModels DeepAgents Service](../services/deepagents/README.md)

