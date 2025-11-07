# Unified Search Service Integration: Framework, Plot, Stdlib, Runtime

## Overview

This document reviews how the `framework`, `plot`, `stdlib`, and `runtime` services can be integrated into the unified search user journey to create a more comprehensive and intelligent search experience.

## Current State

### Service Status

These services appear to be Git submodules that need initialization:
- `services/framework` - Lang infrastructure framework
- `services/plot` - Data visualization service
- `services/stdlib` - Standard library functions
- `services/runtime` - Graph execution runtime

### Existing Capabilities (from codebase)

1. **Framework** (`infrastructure/third_party/orchestration/`):
   - LLM chains, prompts, memory, tools, agents
   - Go-native LangChain-like framework
   - Currently using stubs in graph service

2. **Plot** (scattered across codebase):
   - `scripts/visualize_graph.py` - Neo4j graph visualization
   - `infrastructure/third_party/langextract/benchmarks/plotting.py` - Benchmark visualization
   - NetworkX and matplotlib-based plotting

3. **Runtime** (`services/graph/pkg/graph/runtime.go`):
   - Graph execution with parallelism
   - Checkpointing and state management
   - Execution modes (async, synchronous)
   - Retry logic and error handling

4. **Stdlib** (`pkg/methods/`):
   - ARC transform functions
   - Mathematical operations
   - Standard utility functions

## Integration into Unified Search User Journey

### 1. Framework Integration

#### Purpose
The framework service provides LLM orchestration capabilities that can enhance search results with intelligent processing, summarization, and context enrichment.

#### Integration Points

**A. Search Result Enrichment**
```go
// After unified search returns results
searchResults := unifiedSearch(query)

// Use framework to enrich results
enrichedResults := framework.EnrichSearchResults(
    results: searchResults,
    operations: []string{
        "summarize",
        "extract_key_entities",
        "generate_insights",
    },
)
```

**B. Query Understanding**
```go
// Before executing search
understoodQuery := framework.UnderstandQuery(
    rawQuery: userQuery,
    context: userContext,
    operations: []string{
        "intent_classification",
        "entity_extraction",
        "query_expansion",
    },
)

// Use understood query for better search
searchResults := unifiedSearch(understoodQuery.ExpandedQuery)
```

**C. Multi-Step Search Orchestration**
```go
// Framework orchestrates complex search workflows
workflow := framework.NewWorkflow()
workflow.AddStep("initial_search", unifiedSearch)
workflow.AddStep("refine_results", framework.RefineResults)
workflow.AddStep("generate_summary", framework.Summarize)
workflow.AddStep("extract_insights", framework.ExtractInsights)

results := workflow.Execute(userQuery)
```

#### User Journey Enhancement

1. **Query Input** → Framework analyzes intent
2. **Search Execution** → Unified search with enriched query
3. **Result Processing** → Framework enriches results
4. **Response Generation** → Framework generates intelligent summary

### 2. Plot Integration

#### Purpose
Visualize search results, relationships, and patterns to help users understand complex information.

#### Integration Points

**A. Search Result Visualization**
```python
# After search returns results
searchResults = unified_search(query)

# Generate visualization
plot_service.visualize_search_results(
    results=searchResults,
    visualization_type="network_graph",  # or "timeline", "heatmap", "tree"
    output_format="svg",  # or "png", "interactive_html"
)
```

**B. Knowledge Graph Visualization**
```python
# When search includes knowledge graph results
kgResults = searchResults.get("knowledge_graph", [])

# Visualize relationships
plot_service.visualize_knowledge_graph(
    nodes=kgResults.nodes,
    edges=kgResults.edges,
    highlight=searchResults.query_entities,
    layout="force_directed",  # or "hierarchical", "circular"
)
```

**C. Search Analytics Visualization**
```python
# Visualize search patterns over time
plot_service.visualize_search_analytics(
    metrics=searchAnalytics,
    charts=[
        "query_frequency",
        "result_relevance",
        "source_distribution",
        "temporal_trends",
    ],
)
```

#### User Journey Enhancement

1. **Search Execution** → Results returned
2. **Visualization Generation** → Plot service creates visualizations
3. **Interactive Display** → Users explore results visually
4. **Relationship Discovery** → Visual graphs reveal connections

### 3. Stdlib Integration

#### Purpose
Provide standard utility functions for search result processing, filtering, sorting, and transformation.

#### Integration Points

**A. Result Processing**
```go
// Standard library functions for result manipulation
processedResults := stdlib.ProcessSearchResults(
    results: searchResults,
    operations: []stdlib.Operation{
        stdlib.SortByRelevance,
        stdlib.FilterBySource,
        stdlib.Deduplicate,
        stdlib.TruncateContent(maxLength: 200),
    },
)
```

**B. Query Transformation**
```go
// Standard query transformations
transformedQuery := stdlib.TransformQuery(
    query: userQuery,
    transformations: []stdlib.Transform{
        stdlib.Normalize,
        stdlib.ExpandSynonyms,
        stdlib.RemoveStopWords,
    },
)
```

**C. Data Formatting**
```go
// Standard formatting functions
formattedResults := stdlib.FormatSearchResults(
    results: searchResults,
    format: stdlib.FormatJSON,  // or FormatXML, FormatMarkdown
    options: stdlib.FormatOptions{
        IncludeMetadata: true,
        IncludeSimilarity: true,
        IncludeCitations: true,
    },
)
```

#### User Journey Enhancement

1. **Query Processing** → Stdlib normalizes and transforms
2. **Result Processing** → Stdlib filters, sorts, deduplicates
3. **Response Formatting** → Stdlib formats for display
4. **Consistent Experience** → Standard functions ensure quality

### 4. Runtime Integration

#### Purpose
Execute complex search workflows with parallelism, checkpointing, and state management.

#### Integration Points

**A. Parallel Search Execution**
```go
// Runtime executes multiple searches in parallel
runtime := graph.NewRuntime(
    parallelism: 4,
    executionMode: graph.ExecutionModeAsync,
)

// Execute searches across multiple sources simultaneously
results := runtime.ExecuteParallel([]graph.Node{
    searchInferenceNode,
    knowledgeGraphNode,
    catalogSearchNode,
    perplexityNode,
}, userQuery)
```

**B. Stateful Search Workflows**
```go
// Runtime manages state across search steps
workflow := runtime.NewWorkflow()
workflow.AddNode("initial_search", unifiedSearch)
workflow.AddNode("refine_query", refineQuery)
workflow.AddNode("deep_search", deepSearch)
workflow.AddNode("aggregate", aggregateResults)

// Runtime manages state and checkpoints
results := workflow.ExecuteWithCheckpoint(
    input: userQuery,
    checkpoint: "search_session_123",
)
```

**C. Error Handling and Retry**
```go
// Runtime provides robust error handling
runtime := graph.NewRuntime(
    retries: 3,
    retryDelay: time.Second * 2,
    timeout: time.Second * 30,
)

// Automatic retry on failure
results := runtime.ExecuteWithRetry(
    node: unifiedSearchNode,
    input: userQuery,
)
```

#### User Journey Enhancement

1. **Workflow Definition** → Runtime defines search workflow
2. **Parallel Execution** → Multiple searches run simultaneously
3. **State Management** → Runtime tracks progress and state
4. **Error Recovery** → Automatic retry and error handling

## Complete Integrated User Journey

### Phase 1: Query Understanding (Framework + Stdlib)

```
User Query: "Find customer data in SGMI"
    ↓
Framework: Understand intent, extract entities
    ↓
Stdlib: Normalize, expand synonyms
    ↓
Enriched Query: {
    intent: "data_discovery",
    entities: ["customer", "SGMI"],
    expanded: ["customer", "client", "SGMI", "system"]
}
```

### Phase 2: Parallel Search Execution (Runtime)

```
Enriched Query
    ↓
Runtime: Execute parallel searches
    ├─→ Search Inference Service
    ├─→ Knowledge Graph Search
    ├─→ Catalog Semantic Search
    └─→ Perplexity Web Search (optional)
    ↓
Aggregated Results
```

### Phase 3: Result Processing (Framework + Stdlib)

```
Aggregated Results
    ↓
Framework: Enrich with LLM
    ├─→ Summarize results
    ├─→ Extract key insights
    └─→ Generate recommendations
    ↓
Stdlib: Process and format
    ├─→ Sort by relevance
    ├─→ Deduplicate
    ├─→ Filter by source
    └─→ Format for display
    ↓
Processed Results
```

### Phase 4: Visualization (Plot)

```
Processed Results
    ↓
Plot Service: Generate visualizations
    ├─→ Network graph of relationships
    ├─→ Timeline of results
    ├─→ Source distribution chart
    └─→ Relevance heatmap
    ↓
Visualized Results
```

### Phase 5: Response Delivery (Runtime State Management)

```
Visualized Results
    ↓
Runtime: Manage session state
    ├─→ Checkpoint results
    ├─→ Track user interactions
    └─→ Enable result refinement
    ↓
Final Response to User
```

## Implementation Plan

### Step 1: Initialize Services

```bash
# Initialize submodules
git submodule update --init services/framework
git submodule update --init services/plot
git submodule update --init services/stdlib
git submodule update --init services/runtime
```

### Step 2: Gateway Integration

Add new endpoints to `services/gateway/main.py`:

```python
@app.post("/search/enriched")
async def enriched_search(payload: Dict[str, Any]):
    """Search with framework enrichment."""
    # 1. Framework understands query
    # 2. Unified search executes
    # 3. Framework enriches results
    # 4. Return enriched results

@app.post("/search/visualize")
async def visualize_search(payload: Dict[str, Any]):
    """Search with visualization."""
    # 1. Execute unified search
    # 2. Plot service generates visualizations
    # 3. Return results + visualizations

@app.post("/search/workflow")
async def workflow_search(payload: Dict[str, Any]):
    """Search with runtime workflow."""
    # 1. Runtime defines workflow
    # 2. Execute parallel searches
    # 3. Manage state and checkpoints
    # 4. Return workflow results
```

### Step 3: UI Integration

Update `services/browser/shell/ui/src/modules/Search/SearchModule.tsx`:

```typescript
// Add visualization tab
<Tabs>
  <Tab label="Results" />
  <Tab label="Visualization" />
  <Tab label="Insights" />
</Tabs>

// Add enriched search option
<FormControlLabel
  control={<Checkbox />}
  label="Enable AI enrichment"
/>

// Add workflow search option
<FormControlLabel
  control={<Checkbox />}
  label="Use advanced workflow"
/>
```

### Step 4: Service Configuration

Add to `docker-compose.yml`:

```yaml
services:
  framework:
    build: ./services/framework
    ports:
      - "9005:8080"
    environment:
      - LOCALAI_URL=http://localai:8080
  
  plot:
    build: ./services/plot
    ports:
      - "9006:8080"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
  
  stdlib:
    build: ./services/stdlib
    ports:
      - "9007:8080"
  
  runtime:
    build: ./services/runtime
    ports:
      - "9008:8080"
    environment:
      - GRAPH_SERVICE_URL=http://graph-service:8081
```

## Benefits

### 1. Enhanced Search Intelligence
- Framework provides LLM-powered query understanding and result enrichment
- Better intent classification and entity extraction
- Intelligent summarization and insight generation

### 2. Visual Understanding
- Plot service visualizes complex relationships
- Interactive graphs help users explore connections
- Analytics charts reveal search patterns

### 3. Consistent Processing
- Stdlib ensures standardized result processing
- Reliable filtering, sorting, and formatting
- Consistent user experience

### 4. Robust Execution
- Runtime provides parallel execution and error handling
- State management enables complex workflows
- Checkpointing supports long-running searches

## Example Use Cases

### Use Case 1: Complex Research Query

**User Query**: "How do customer data flows connect to regulatory reporting?"

**Journey**:
1. Framework extracts entities: ["customer", "data flows", "regulatory reporting"]
2. Runtime executes parallel searches across all sources
3. Framework generates research summary
4. Plot visualizes data flow connections
5. User sees comprehensive research report with visualizations

### Use Case 2: Exploratory Search

**User Query**: "Show me everything about SGMI"

**Journey**:
1. Framework expands query to include related terms
2. Runtime executes broad search across all sources
3. Stdlib deduplicates and organizes results
4. Plot creates network graph of relationships
5. User explores interconnected information visually

### Use Case 3: Refinement Workflow

**User Query**: "Find customer tables" → User refines → "Find customer tables with PII"

**Journey**:
1. Initial search executed
2. Runtime checkpoints state
3. User refines query
4. Runtime resumes from checkpoint
5. Framework compares and highlights differences
6. Plot shows refinement impact

## Next Steps

1. **Initialize Submodules**: Set up framework, plot, stdlib, runtime services
2. **Gateway Integration**: Add enriched, visualize, and workflow endpoints
3. **UI Enhancement**: Add visualization and enrichment options
4. **Testing**: Test integrated user journey end-to-end
5. **Documentation**: Update user guides with new capabilities

## Conclusion

Integrating framework, plot, stdlib, and runtime services into the unified search user journey creates a comprehensive, intelligent search experience that:
- Understands user intent better
- Executes searches more efficiently
- Processes results more intelligently
- Visualizes information more effectively
- Manages complex workflows robustly

This integration transforms search from a simple query-response mechanism into an intelligent research and discovery platform.

