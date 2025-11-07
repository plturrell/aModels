# Search Integration Architecture

## Overview

The Search module in the browser UI connects to a broader ecosystem that includes **Goose** (database migrations and AI agent framework) and **Deep Research** (autonomous research service). This document explains how these components integrate.

## Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser UI (Search Module)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SearchModule → /search/v1/search                     │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Shell Server (Proxy Layer)                      │
│  /search/* → search-inference service                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Search       │ │ Extract      │ │ Catalog      │
│ Inference    │ │ Service      │ │ Service      │
│ Service      │ │              │ │              │
│              │ │              │ │              │
│ /v1/search   │ │ /knowledge-  │ │ /catalog/    │
│              │ │ graph/search │ │ semantic-    │
│              │ │              │ │ search      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                 │
       └────────────────┼─────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │     Knowledge Graph (Neo4j)   │
        │     + Vector Store (Elastic)   │
        └───────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Deep         │ │ Goose        │ │ Autonomous   │
│ Research     │ │ Migrations   │ │ Intelligence │
│              │ │              │ │ Layer        │
└──────────────┘ └──────────────┘ └──────────────┘
```

## 1. Search → Deep Research Integration

### How They Connect

**Deep Research** uses search as one of its tools for autonomous research:

```python
# services/catalog/research/deep_research_tool.py
payload = {
    "query": query,
    "context": context or {},
    "tools": ["sparql_query", "catalog_search"],  # Search is a tool!
}
```

**Catalog Search Tool** (`search_catalog_semantic`):
```python
# services/catalog/research/deep_research_tool.py:68
async def search_catalog_semantic(query: str, filters: Optional[Dict[str, Any]] = None):
    """
    Semantic search in catalog.
    Used by Open Deep Research as a tool.
    """
    response = await client.post(
        CATALOG_SPARQL_URL.replace("/sparql", "/semantic-search"),
        json={"query": query, **(filters or {})}
    )
```

**Deep Research Client** (Go):
```go
// services/catalog/research/client.go:152
req := &ResearchRequest{
    Query: query,
    Tools: []string{"sparql_query", "catalog_search"},  // Search tool
}
```

### Integration Flow

1. **User Query** → Browser Search Module
2. **Search Service** → Returns semantic search results
3. **Deep Research** → Can use search results as context for research
4. **Research Report** → Generated using search findings

### Use Cases

- **Metadata Research**: Deep Research queries search to find relevant data elements
- **Context Building**: Search results provide context for research questions
- **Lineage Discovery**: Search helps find related entities for research reports

### Code References

- `services/catalog/research/deep_research_tool.py` - Deep Research tool that uses search
- `services/catalog/autonomous/intelligence_layer.go:286` - Deep Research called in autonomous tasks
- `services/catalog/workflows/unified_integration.go:350` - Research report generation

## 2. Search → Goose Integration

### Two Types of Goose

#### A. Goose Migrations (Database)

**Purpose**: Database schema management for search-related tables

**Integration**:
- Search results can be stored in database tables
- Goose migrations manage schema for:
  - Search history
  - Search analytics
  - User preferences
  - Search indexes

**Code References**:
- `services/catalog/autonomous/integration.go:98` - `RunGooseMigration` for autonomous tasks
- `services/catalog/main.go:142` - Goose migrations run on catalog startup

#### B. Goose AI Agent (MCP Framework)

**Purpose**: AI agent that can use search as a tool via Model Context Protocol

**Integration**:
- Goose agents can query search service
- Search results feed into agent context
- Agents can perform multi-step research using search

**Code References**:
- `infrastructure/third_party/goose/` - Goose AI agent framework
- `services/catalog/autonomous/intelligence_layer.go:23` - `gooseEnabled` flag
- `services/catalog/autonomous/integration.go:12` - Integrated autonomous system

### Integration Flow

```
Goose Agent → MCP Protocol → Search Tool → Search Service → Results → Agent Context
```

## 3. Search → Knowledge Graph Integration

### Direct Integration

The **Extract Service** provides knowledge graph search:

**Endpoint**: `POST /knowledge-graph/search`

**Features**:
- Semantic search across Neo4j knowledge graph
- Vector similarity search
- Hybrid search (keyword + semantic)

**Code References**:
- `services/extract/main.go:412` - `handleVectorSearch` endpoint
- `services/catalog/breakdetection/search.go:60` - Uses knowledge graph search
- `services/extract/INTEGRATION.md:208` - Knowledge graph search documentation

### Search Service → Knowledge Graph

The search-inference service can:
- Index knowledge graph entities
- Search across graph nodes and relationships
- Return graph-structured results

## 4. Autonomous Intelligence Layer Integration

### Complete Flow

The **Autonomous Intelligence Layer** orchestrates all components:

```go
// services/catalog/autonomous/intelligence_layer.go:264
func (il *IntelligenceLayer) ExecuteAutonomousTask(...) {
    // Step 1: Use Deep Research to understand context
    researchReport, err := il.deepResearchClient.ResearchMetadata(...)
    
    // Step 2: Use DeepAgents to plan and decompose task
    plan, err := il.planWithDeepAgents(ctx, task, researchResult)
    
    // Step 3: Execute plan using Unified Workflow
    result, err := il.executePlanWithUnifiedWorkflow(ctx, plan, task)
    
    // Step 4: Learn from execution (Goose migration records this)
    il.learnFromExecution(task, result, err)
}
```

### Search's Role

1. **Context Discovery**: Search finds relevant documents/entities
2. **Research Support**: Deep Research uses search results
3. **Knowledge Building**: Results feed into knowledge graph
4. **Learning**: Search patterns inform autonomous learning

## 5. Practical Integration Examples

### Example 1: Deep Research Using Search

```python
# Deep Research tool uses search
async def research_metadata(query: str):
    # Search finds relevant catalog entries
    search_results = await search_catalog_semantic(query)
    
    # Deep Research uses results for context
    research_report = await deep_research.research(
        query=query,
        context={"search_results": search_results}
    )
```

### Example 2: Autonomous Task with Search

```go
// Autonomous task execution
task := &AutonomousTask{
    Query: "Find all customer data elements",
}

// Step 1: Deep Research uses search internally
researchReport := deepResearchClient.ResearchMetadata(...)

// Step 2: Search results inform planning
plan := planWithDeepAgents(researchReport)

// Step 3: Execute with search context
result := executePlanWithUnifiedWorkflow(plan)

// Step 4: Record in database via Goose migration
recordExecutionInDB(task, result)
```

### Example 3: Break Detection with Search

```go
// services/catalog/breakdetection/search.go
func (bss *BreakSearchService) SearchSimilarBreaks(...) {
    // Search knowledge graph for similar breaks
    searchRequest := map[string]interface{}{
        "query": buildBreakSearchQuery(breakRecord),
        "artifact_type": "break",
        "use_semantic": true,
        "use_hybrid_search": true,
    }
    
    // Call extract service knowledge graph search
    searchURL := fmt.Sprintf("%s/knowledge-graph/search", ...)
}
```

## 6. Configuration

### Environment Variables

**Search Service**:
```bash
SHELL_SEARCH_ENDPOINT=http://localhost:8090  # Browser UI
SEARCH_CONFIG=/workspace/config/search-inference.yaml
```

**Deep Research**:
```bash
DEEP_RESEARCH_URL=http://localhost:8085
CATALOG_SPARQL_URL=http://localhost:8084/catalog/sparql
```

**Goose**:
```bash
# Database connection for migrations
DMS_POSTGRES_DSN=postgresql+psycopg://...
```

### Service URLs

- **Search Inference**: `http://search-inference:8090`
- **Extract Service (KG Search)**: `http://extract-service:19080`
- **Deep Research**: `http://deep-research:2024` or `http://localhost:8085`
- **Catalog (Semantic Search)**: `http://catalog:8084/catalog/semantic-search`

## 7. Future Enhancements

### Potential Integrations

1. **Search → Deep Research Direct Integration**
   - Add search results directly to Deep Research context
   - Enable Deep Research to query search service via API

2. **Search → Goose Agent Integration**
   - Create MCP server for search service
   - Enable Goose agents to use search as a tool

3. **Unified Search Interface**
   - Combine search-inference, knowledge graph search, and catalog search
   - Single search endpoint that routes to appropriate backend

4. **Search Analytics via Goose**
   - Store search patterns in database
   - Use Goose migrations for search analytics schema
   - Feed analytics into autonomous learning

## 8. Summary

### Search Module Connections

1. **Direct**: Search Module → Search Inference Service
2. **Via Knowledge Graph**: Search → Extract Service → Neo4j/Elasticsearch
3. **Via Deep Research**: Search → Deep Research (as tool) → Research Reports
4. **Via Goose**: Search → Database (migrations) + AI Agent (MCP tools)
5. **Via Autonomous Layer**: Search → All components orchestrated together

### Key Integration Points

- **Deep Research** uses search as a research tool
- **Goose Migrations** manage search-related database schema
- **Goose AI Agent** can use search via MCP protocol
- **Knowledge Graph** provides semantic search capabilities
- **Autonomous Intelligence** orchestrates search with other services

The Search module is not isolated—it's part of a comprehensive intelligence ecosystem that enables autonomous research, learning, and knowledge discovery.

