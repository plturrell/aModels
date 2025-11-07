# DeepAgents Service Integration Guide

## Overview

The DeepAgents service provides a general-purpose tool-using agent that integrates with all other aModels services. It acts as a coordination layer that can orchestrate complex workflows across Knowledge Graphs, AgentFlow, and Orchestration services.

## Service Information

- **Port**: 9004
- **Technology**: Python, FastAPI, deepagents library
- **Base URL**: `http://deepagents-service:9004`

## Integration Points

### 1. Knowledge Graph Integration

**Tool**: `query_knowledge_graph`

**Purpose**: Query the Neo4j knowledge graph using Cypher queries.

**Configuration**:
```bash
EXTRACT_SERVICE_URL=http://extract-service:19080
```

**Usage Example**:
```python
# The agent automatically uses this tool when needed
# Example agent conversation:
{
  "messages": [
    {
      "role": "user",
      "content": "Find all tables in the SGMI project and their column counts"
    }
  ]
}
```

**Tool Implementation**: `services/deepagents/tools/knowledge_graph_tool.py`

**API Endpoint Used**: `POST {EXTRACT_SERVICE_URL}/knowledge-graph/query`

**Input Parameters**:
- `query` (string): Cypher query to execute
- `project_id` (optional): Filter by project ID
- `system_id` (optional): Filter by system ID

**Output**: Formatted query results as readable text

---

### 2. AgentFlow Integration

**Tool**: `run_agentflow_flow`

**Purpose**: Execute pre-configured LangFlow flows.

**Configuration**:
```bash
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
```

**Usage Example**:
```python
# Agent automatically invokes flows when needed
{
  "messages": [
    {
      "role": "user",
      "content": "Run the SGMI pipeline reconciliation flow"
    }
  ]
}
```

**Tool Implementation**: `services/deepagents/tools/agentflow_tool.py`

**API Endpoint Used**: `POST {AGENTFLOW_SERVICE_URL}/run`

**Input Parameters**:
- `flow_id` (string): Flow identifier
- `inputs` (dict): Flow input parameters

**Output**: Flow execution results

---

### 3. Orchestration Integration

**Tool**: `run_orchestration_chain`

**Purpose**: Execute orchestration chains for analysis, Q&A, and summarization.

**Configuration**:
```bash
GRAPH_SERVICE_URL=http://graph-service:8081
```

**Usage Example**:
```python
# Agent uses orchestration chains for analysis
{
  "messages": [
    {
      "role": "user",
      "content": "Analyze the data quality of the SGMI system using orchestration chains"
    }
  ]
}
```

**Tool Implementation**: `services/deepagents/tools/orchestration_tool.py`

**API Endpoint Used**: `POST {GRAPH_SERVICE_URL}/orchestration/process`

**Supported Chain Types**:
- `llm_chain`: Basic LLM chain
- `question_answering`: Context-aware Q&A
- `summarization`: Text summarization
- `knowledge_graph_analyzer`: Analyze knowledge graphs
- `data_quality_analyzer`: Analyze data quality metrics
- `pipeline_analyzer`: Analyze data pipelines
- `sql_analyzer`: Analyze SQL queries

**Input Parameters**:
- `chain_name` (string): Chain type to execute
- `inputs` (dict): Chain input parameters
- `knowledge_graph_query` (optional): Cypher query to enrich context

**Output**: Chain execution results as text

---

## API Endpoints

### Health Check

**Endpoint**: `GET /healthz`

**Response**: HTTP 200 OK

**Usage**:
```bash
curl http://deepagents-service:9004/healthz
```

---

### Invoke Agent

**Endpoint**: `POST /invoke`

**Request Body**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Your question or task here"
    }
  ],
  "stream": false,
  "config": {}
}
```

**Response**:
```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "Agent response with tool calls and results"
    }
  ],
  "result": {
    "tool_calls": [...],
    "execution_time": 1.23
  }
}
```

**Example**:
```python
import httpx

client = httpx.Client()
response = client.post(
    "http://deepagents-service:9004/invoke",
    json={
        "messages": [
            {"role": "user", "content": "Analyze the SGMI pipeline"}
        ]
    }
)
result = response.json()
```

---

### Stream Responses

**Endpoint**: `POST /stream`

**Request Body**: Same as `/invoke`

**Response**: Server-Sent Events (SSE) stream

**Usage**:
```python
import httpx

with httpx.stream(
    "POST",
    "http://deepagents-service:9004/stream",
    json={"messages": [{"role": "user", "content": "..."}]}
) as response:
    for line in response.iter_lines():
        print(line)
```

---

### Get Agent Info

**Endpoint**: `GET /agent/info`

**Response**:
```json
{
  "agent_type": "deepagents",
  "tools": [
    "query_knowledge_graph",
    "run_agentflow_flow",
    "run_orchestration_chain",
    "write_todos",
    "task",
    "ls",
    "read_file",
    "write_file"
  ],
  "model": "anthropic/claude-3-5-sonnet"
}
```

---

## Integration Examples

### Example 1: Complete Pipeline Analysis

```python
import httpx

client = httpx.Client(timeout=300.0)

response = client.post(
    "http://deepagents-service:9004/invoke",
    json={
        "messages": [
            {
                "role": "user",
                "content": """Analyze the SGMI pipeline end-to-end:
                1. Query knowledge graph for all Control-M jobs
                2. Find associated SQL queries and tables
                3. Analyze data flow and dependencies
                4. Use orchestration chain to generate optimization recommendations
                5. Create a summary report"""
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
    "http://deepagents-service:9004/invoke",
    json={
        "messages": [
            {
                "role": "user",
                "content": """Assess data quality for the SGMI system:
                1. Query knowledge graph for all tables
                2. For each table, analyze data quality metrics
                3. Use orchestration chain (data_quality_analyzer) to generate recommendations
                4. Create a prioritized action plan"""
            }
        ]
    }
)
```

### Example 3: Automated Flow Generation

```python
response = client.post(
    "http://deepagents-service:9004/invoke",
    json={
        "messages": [
            {
                "role": "user",
                "content": """Create an AgentFlow flow for the Control-M pipeline:
                1. Query knowledge graph for Control-M jobs and SQL queries
                2. Analyze dependencies and data flow
                3. Generate LangFlow flow JSON structure
                4. Use AgentFlow tool to create the flow"""
            }
        ]
    }
)
```

---

## Configuration

### Required Environment Variables

```bash
# Service URLs (required)
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081

# Model Configuration (at least one required)
ANTHROPIC_API_KEY=your_key  # Preferred
OPENAI_API_KEY=your_key     # Alternative
LOCALAI_URL=http://localai:8080  # Fallback

# Service Port (optional, default: 9004)
DEEPAGENTS_PORT=9004
```

### Optional Configuration

```bash
# Enable/disable service (default: enabled)
DEEPAGENTS_ENABLED=true
```

---

## Error Handling

The DeepAgents service implements graceful error handling:

1. **Tool Failures**: If a tool call fails, the agent receives an error message and can retry or use alternative approaches
2. **Service Unavailability**: If downstream services are unavailable, tools return error messages that the agent can handle
3. **Timeout**: Default timeout is 120 seconds for agent operations, 30-60 seconds for tool calls

**Error Response Format**:
```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "Error: Failed to query knowledge graph: HTTP 503 - Service unavailable"
    }
  ],
  "result": {
    "error": "Tool execution failed",
    "tool": "query_knowledge_graph"
  }
}
```

---

## Integration with Other Services

### From Extract Service

The extract service can call DeepAgents for graph analysis:

```go
deepAgentsClient := NewDeepAgentsClient(logger)
response, err := deepAgentsClient.AnalyzeKnowledgeGraph(ctx, graphSummary, projectID, systemID)
```

**File**: `services/extract/deepagents.go`

### From Graph Service

The graph service can include DeepAgents in unified workflows:

```go
deepAgentNode := RunDeepAgentNode(opts.DeepAgentsServiceURL)
result, err := deepAgentNode(ctx, state)
```

**File**: `services/graph/pkg/workflows/deepagents_processor.go`

---

## Best Practices

1. **Use Clear Instructions**: Provide specific, actionable instructions to the agent
2. **Break Down Complex Tasks**: The agent can use `write_todos` to plan complex workflows
3. **Leverage Tools**: The agent automatically selects appropriate tools based on the task
4. **Handle Errors Gracefully**: The agent can retry failed operations or use alternatives
5. **Monitor Tool Usage**: Check agent responses to see which tools were used

---

## Troubleshooting

### Service Not Responding

1. Check health endpoint: `curl http://deepagents-service:9004/healthz`
2. Verify environment variables are set correctly
3. Check service logs for errors

### Tool Failures

1. Verify downstream services are running and accessible
2. Check service URLs in environment variables
3. Review tool error messages in agent responses

### Model Issues

1. Verify API keys are set correctly
2. Check model availability (Anthropic, OpenAI, or LocalAI)
3. Review rate limits and quotas

---

## References

- [DeepAgents README](./README.md)
- [Lang Infrastructure Review](../../docs/lang-infrastructure-review.md)
- [deepagents PyPI](https://pypi.org/project/deepagents/)

