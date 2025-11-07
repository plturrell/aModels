# Extract Service Integration Guide

## Overview

The Extract service provides structured data extraction using LangExtract, knowledge graph persistence, and integration with orchestration chains and DeepAgents for analysis.

## Service Information

- **Port**: 19080 (HTTP), 9090 (gRPC), 8815 (Arrow Flight)
- **Technology**: Go
- **Base URL**: `http://extract-service:19080`

## Integration Points

### 1. LangExtract Integration

**Purpose**: Structured entity extraction from documents.

**File**: `services/extract/extract_logic.go`

**Configuration**:
```bash
LANGEXTRACT_URL=http://langextract-service:port
LANGEXTRACT_API_KEY=your_key
```

**Usage**:
```go
// Extract entities from text
req := extractRequest{
    Document: "Text to extract from",
    PromptDescription: "Extract people, dates, and locations",
    Examples: []Example{...},
    ModelID: "gemini-2.5-flash",
}

resp, err := server.runExtract(ctx, req)
```

**API Endpoint**: `POST /extract`

**Request**:
```json
{
  "document": "Text content",
  "prompt_description": "Extraction instructions",
  "examples": [...],
  "model_id": "gemini-2.5-flash"
}
```

**Response**:
```json
{
  "entities": {
    "people": ["John Doe"],
    "dates": ["2024-01-01"],
    "locations": ["New York"]
  },
  "extractions": [...]
}
```

---

### 2. Orchestration Integration

**Purpose**: Route extraction tasks to appropriate orchestration chains.

**File**: `services/extract/orchestration_integration.go`

**Component**: `OrchestrationChainMatcher`

**Usage**:
```go
matcher := NewOrchestrationChainMatcher(logger)
matcher.SetExtractServiceURL(extractServiceURL)

chainName, score, err := matcher.MatchChainToTask(
    "Process transaction data",
    "transactions",
    "transaction",
)
```

**Chain Types**:
- `transaction_processing_chain`: For transaction tables
- `reference_lookup_chain`: For reference data
- `staging_etl_chain`: For staging tables
- `test_processing_chain`: For test data
- `default_chain`: Fallback

**Configuration**:
```bash
USE_SAP_RPT_EMBEDDINGS=true  # Enable semantic matching
```

---

### 3. DeepAgents Integration

**Purpose**: AI-powered graph analysis and insights.

**File**: `services/extract/deepagents.go`

**Component**: `DeepAgentsClient`

**Usage**:
```go
client := NewDeepAgentsClient(logger)

summary := FormatGraphSummary(nodes, edges, quality, metrics)
response, err := client.AnalyzeKnowledgeGraph(ctx, summary, projectID, systemID)
```

**Features**:
- Non-fatal integration (graceful degradation)
- Health check before attempting analysis
- Retry logic with exponential backoff
- Automatic timeout handling

**Configuration**:
```bash
DEEPAGENTS_URL=http://deepagents-service:9004
DEEPAGENTS_ENABLED=true  # Default: enabled
```

---

## API Endpoints

### Health Check

**Endpoint**: `GET /healthz`

**Response**:
```json
{
  "status": "ok",
  "service": "extract",
  "langextract": "http://langextract:port",
  "deepagents": "enabled"
}
```

---

### Extract Entities

**Endpoint**: `POST /extract`

**Request**:
```json
{
  "document": "Text to extract from",
  "prompt_description": "Extract entities",
  "examples": [
    {
      "text": "Example text",
      "extractions": [...]
    }
  ],
  "model_id": "gemini-2.5-flash"
}
```

**Response**:
```json
{
  "entities": {
    "people": ["..."],
    "dates": ["..."],
    "locations": ["..."]
  },
  "extractions": [...]
}
```

---

### Query Knowledge Graph

**Endpoint**: `POST /knowledge-graph/query`

**Request**:
```json
{
  "query": "MATCH (n:Table) RETURN n LIMIT 10",
  "params": {
    "project_id": "sgmi"
  }
}
```

**Response**:
```json
{
  "columns": ["name", "type"],
  "data": [
    {"name": "table1", "type": "fact"},
    {"name": "table2", "type": "dimension"}
  ]
}
```

---

### Semantic Search

**Endpoint**: `POST /knowledge-graph/search`

**Request**:
```json
{
  "query": "transaction tables",
  "artifact_type": "table",
  "limit": 10,
  "use_semantic": true,
  "use_hybrid_search": true
}
```

**Response**:
```json
{
  "results": [
    {
      "metadata": {...},
      "score": 0.95
    }
  ]
}
```

---

## Integration Examples

### Example 1: Extract and Analyze

```go
// Extract entities
extractReq := extractRequest{
    Document: documentText,
    PromptDescription: "Extract regulatory requirements",
    ModelID: "gemini-2.5-flash",
}
extractResp, err := server.runExtract(ctx, extractReq)

// Analyze with DeepAgents
summary := FormatGraphSummary(nodes, edges, quality, metrics)
analysis, err := deepAgentsClient.AnalyzeKnowledgeGraph(ctx, summary, projectID, systemID)
```

### Example 2: Route to Orchestration Chain

```go
matcher := NewOrchestrationChainMatcher(logger)
matcher.SetExtractServiceURL(extractServiceURL)

chainName, score, err := matcher.MatchChainToTask(
    "Analyze data quality",
    "quality_metrics",
    "reference",
)

// Use chainName to route to appropriate orchestration chain
```

### Example 3: Query Knowledge Graph

```python
import httpx

client = httpx.Client()

response = client.post(
    "http://extract-service:19080/knowledge-graph/query",
    json={
        "query": "MATCH (n:Table {project_id: $project_id}) RETURN n",
        "params": {"project_id": "sgmi"}
    }
)

result = response.json()
```

---

## Configuration

### Required Environment Variables

```bash
# LangExtract
LANGEXTRACT_URL=http://langextract-service:port
LANGEXTRACT_API_KEY=your_key

# Knowledge Graph (Neo4j)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# DeepAgents (optional)
DEEPAGENTS_URL=http://deepagents-service:9004
DEEPAGENTS_ENABLED=true  # Default: enabled

# LocalAI (for domain detection)
LOCALAI_URL=http://localai:8080
```

### Optional Configuration

```bash
# Orchestration integration
USE_SAP_RPT_EMBEDDINGS=true  # Enable semantic chain matching

# Persistence
SQLITE_PATH=/data/extract.db
REDIS_ADDR=redis:6379
REDIS_PASSWORD=
REDIS_DB=0

# Training
TRAINING_OUTPUT_DIR=/data/training
```

---

## Error Handling

### LangExtract Errors

The service handles LangExtract errors gracefully:

```go
resp, err := server.invokeLangextract(ctx, payload)
if err != nil {
    return extractResponse{}, &extractError{
        status: http.StatusBadGateway,
        err:    fmt.Errorf("langextract call failed: %w", err),
    }
}
```

### DeepAgents Errors

DeepAgents integration is non-fatal:

```go
response, err := client.AnalyzeKnowledgeGraph(ctx, summary, projectID, systemID)
if err != nil {
    // Log error but continue
    logger.Printf("DeepAgents analysis failed: %v", err)
    return nil, nil  // Non-fatal
}
```

**Features**:
- Health check before attempting analysis
- Automatic retry with exponential backoff
- Graceful degradation if service unavailable
- Timeout handling (5s health check, 120s request)

---

## Integration with Other Services

### From Graph Service

The graph service calls extract for knowledge graph processing:

**File**: `services/graph/pkg/workflows/knowledge_graph_processor.go`

```go
kgNode := ProcessKnowledgeGraphNode(extractServiceURL)
result, err := kgNode(ctx, state)
```

### From DeepAgents

DeepAgents queries the knowledge graph via extract service:

**File**: `services/deepagents/tools/knowledge_graph_tool.py`

```python
response = client.post(
    f"{EXTRACT_SERVICE_URL}/knowledge-graph/query",
    json={"query": cypher_query}
)
```

---

## Regulatory Extraction

The service supports specialized regulatory extraction:

### MAS 610 Extraction

**File**: `services/extract/regulatory/mas_610.go`

```go
extractor := NewMAS610Extractor(baseExtractor, logger)
result, err := extractor.ExtractMAS610(ctx, document, source, version, user)
```

### BCBS 239 Extraction

**File**: `services/extract/regulatory/bcbs239.go`

```go
extractor := NewBCBS239Extractor(baseExtractor, logger)
result, err := extractor.ExtractBCBS239(ctx, document, source, version, user)
```

---

## Best Practices

1. **Use Appropriate Models**: Choose model based on task complexity
   - `gemini-2.5-flash`: Fast, cost-effective (default)
   - `gemini-2.5-pro`: Complex reasoning tasks

2. **Provide Good Examples**: High-quality few-shot examples improve extraction accuracy

3. **Handle Errors Gracefully**: DeepAgents integration is non-fatal, handle accordingly

4. **Use Semantic Search**: Enable `USE_SAP_RPT_EMBEDDINGS` for better chain matching

5. **Monitor Performance**: Track extraction times and success rates

---

## Troubleshooting

### LangExtract Failures

1. Verify `LANGEXTRACT_URL` is correct
2. Check `LANGEXTRACT_API_KEY` is set
3. Review LangExtract service logs
4. Check model availability

### Knowledge Graph Query Failures

1. Verify Neo4j connection: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
2. Test connection: `curl http://extract-service:19080/healthz`
3. Check Cypher query syntax
4. Review Neo4j logs

### DeepAgents Integration Issues

1. Check `DEEPAGENTS_ENABLED` is not `false`
2. Verify `DEEPAGENTS_URL` is correct
3. Test DeepAgents health: `curl http://deepagents-service:9004/healthz`
4. Review service logs for timeout/connection errors

---

## References

- [Extract Service README](./README.md)
- [LangExtract Documentation](../../infrastructure/third_party/langextract/README.md)

