# Search as Prompt Engine: Dynamic Dashboards and Narratives

## Overview

This document reviews and rates the approach of using **search as a prompt engine** to generate dynamic dashboards and narratives through the framework service. The concept leverages search results as structured input data that the framework (LLM orchestration) processes to create intelligent visualizations and narratives.

## Architecture Concept

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                               │
│              "Show me customer data trends"                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Unified Search (Prompt Engine)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Inference    │  │ Knowledge    │  │ Catalog      │     │
│  │ Service      │  │ Graph        │  │ Search       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  Returns: Structured search results with metadata           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Framework (LLM Orchestration)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Narrative Generation Chain                        │  │
│  │    - Analyze search results                          │  │
│  │    - Extract key insights                            │  │
│  │    - Generate executive summary                      │  │
│  │    - Create detailed narrative                       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. Dashboard Configuration Chain                     │  │
│  │    - Identify data patterns                          │  │
│  │    - Determine visualization types                   │  │
│  │    - Generate dashboard schema                       │  │
│  │    - Create chart configurations                     │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Narrative    │ │ Dashboard    │ │ Visualization│
│ Output       │ │ Config       │ │ Data         │
│              │ │              │ │              │
│ - Summary    │ │ - Chart types│ │ - Processed  │
│ - Insights   │ │ - Metrics    │ │   data       │
│ - Details    │ │ - Layout     │ │ - Aggregates │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Rating: ⭐⭐⭐⭐⭐ (5/5)

### Strengths

1. **Powerful Concept**: Using search as a prompt engine is innovative and leverages existing infrastructure
2. **Dynamic Generation**: Dashboards and narratives adapt to search results automatically
3. **Intelligent Analysis**: Framework LLM chains provide context-aware insights
4. **Scalable**: Works with any search query and result set
5. **User-Friendly**: Users get narratives and dashboards without manual configuration

### Challenges

1. **Latency**: Multiple LLM calls (search → framework → narrative/dashboard) add time
2. **Cost**: LLM processing for every search can be expensive
3. **Consistency**: LLM-generated content may vary between runs
4. **Complexity**: Requires careful prompt engineering for reliable outputs

## Implementation Approach

### Phase 1: Narrative Generation

**Use Case**: Generate intelligent narratives from search results

**Framework Chain**: `narrative_generator`

**Prompt Template**:
```
You are a data analyst. Analyze the following search results and generate a comprehensive narrative.

Search Query: {query}
Number of Results: {result_count}
Sources: {sources}

Search Results:
{formatted_results}

Generate:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis (paragraphs)
4. Recommendations (if applicable)
5. Data Quality Assessment
```

**Implementation**:
```python
async def generate_narrative_from_search(
    search_results: Dict[str, Any],
    query: str
) -> Dict[str, Any]:
    """Generate narrative from search results using framework."""
    
    # Format search results for LLM
    formatted_results = format_search_results_for_llm(search_results)
    
    # Use framework orchestration chain
    orchestration_payload = {
        "orchestration_request": {
            "chain_name": "narrative_generator",
            "inputs": {
                "query": query,
                "result_count": search_results["total_count"],
                "sources": list(search_results["sources"].keys()),
                "formatted_results": formatted_results,
                "metadata": search_results["metadata"]
            }
        }
    }
    
    r = await client.post(
        f"{GRAPH_SERVICE_URL}/orchestration/process",
        json=orchestration_payload,
        timeout=30.0
    )
    
    if r.status_code == 200:
        result = r.json()
        narrative_text = result.get("orchestration_text", "")
        
        # Parse narrative into structured format
        return parse_narrative(narrative_text)
    
    return {"error": "Failed to generate narrative"}
```

### Phase 2: Dynamic Dashboard Generation

**Use Case**: Generate dashboard configurations from search results

**Framework Chain**: `dashboard_generator`

**Prompt Template**:
```
You are a data visualization expert. Analyze the following search results and generate a dashboard configuration.

Search Query: {query}
Search Results: {formatted_results}
Metadata: {metadata}

Generate a JSON dashboard configuration with:
1. Dashboard Title
2. Chart Types (bar, line, pie, table, etc.)
3. Metrics to Display
4. Data Aggregations
5. Layout Configuration
6. Filters and Interactions

Return valid JSON only.
```

**Implementation**:
```python
async def generate_dashboard_from_search(
    search_results: Dict[str, Any],
    query: str
) -> Dict[str, Any]:
    """Generate dashboard configuration from search results."""
    
    # Format search results
    formatted_results = format_search_results_for_llm(search_results)
    
    # Use framework to generate dashboard config
    orchestration_payload = {
        "orchestration_request": {
            "chain_name": "dashboard_generator",
            "inputs": {
                "query": query,
                "formatted_results": formatted_results,
                "metadata": search_results["metadata"],
                "visualization_data": search_results.get("visualization", {})
            }
        }
    }
    
    r = await client.post(
        f"{GRAPH_SERVICE_URL}/orchestration/process",
        json=orchestration_payload,
        timeout=30.0
    )
    
    if r.status_code == 200:
        result = r.json()
        dashboard_json = result.get("orchestration_text", "")
        
        # Parse and validate dashboard config
        dashboard_config = json.loads(dashboard_json)
        return validate_dashboard_config(dashboard_config)
    
    return {"error": "Failed to generate dashboard"}
```

### Phase 3: Combined Narrative + Dashboard

**Use Case**: Generate both narrative and dashboard together

**Implementation**:
```python
async def generate_narrative_and_dashboard(
    search_results: Dict[str, Any],
    query: str
) -> Dict[str, Any]:
    """Generate both narrative and dashboard from search results."""
    
    # Run both in parallel
    narrative_task = generate_narrative_from_search(search_results, query)
    dashboard_task = generate_dashboard_from_search(search_results, query)
    
    narrative, dashboard = await asyncio.gather(
        narrative_task,
        dashboard_task,
        return_exceptions=True
    )
    
    return {
        "narrative": narrative if not isinstance(narrative, Exception) else {"error": str(narrative)},
        "dashboard": dashboard if not isinstance(dashboard, Exception) else {"error": str(dashboard)},
        "query": query,
        "search_metadata": search_results["metadata"]
    }
```

## Framework Chain Definitions

### 1. Narrative Generator Chain

**File**: `services/graph/pkg/workflows/orchestration_processor.go`

```go
case "narrative_generator":
    promptTemplate := prompts.NewPromptTemplate(
        `You are a data analyst. Analyze the following search results and generate a comprehensive narrative.

Search Query: {{.query}}
Number of Results: {{.result_count}}
Sources: {{.sources}}

Search Results:
{{.formatted_results}}

Generate:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis (paragraphs)
4. Recommendations (if applicable)
5. Data Quality Assessment

Format your response as structured JSON:
{
  "executive_summary": "...",
  "key_findings": ["...", "..."],
  "detailed_analysis": "...",
  "recommendations": ["...", "..."],
  "data_quality": "..."
}`,
        []string{"query", "result_count", "sources", "formatted_results"},
    )
    return chains.NewLLMChain(llm, promptTemplate), nil
```

### 2. Dashboard Generator Chain

```go
case "dashboard_generator":
    promptTemplate := prompts.NewPromptTemplate(
        `You are a data visualization expert. Analyze the following search results and generate a dashboard configuration.

Search Query: {{.query}}
Search Results: {{.formatted_results}}
Metadata: {{.metadata}}
Visualization Data: {{.visualization_data}}

Generate a JSON dashboard configuration with:
1. Dashboard Title
2. Chart Types (bar, line, pie, table, etc.)
3. Metrics to Display
4. Data Aggregations
5. Layout Configuration
6. Filters and Interactions

Return ONLY valid JSON, no other text.`,
        []string{"query", "formatted_results", "metadata", "visualization_data"},
    )
    return chains.NewLLMChain(llm, promptTemplate), nil
```

## Gateway Integration

### New Endpoints

```python
@app.post("/search/narrative")
async def generate_search_narrative(payload: Dict[str, Any]) -> Any:
    """
    Generate narrative from search results.
    
    Request:
    {
        "query": "search query",
        "search_results": {...},  // Optional: if not provided, performs search first
        "enable_framework": true
    }
    """
    query = payload.get("query", "")
    search_results = payload.get("search_results")
    
    # If search results not provided, perform search first
    if not search_results:
        search_payload = {
            "query": query,
            "enable_framework": True,
            "enable_plot": True
        }
        search_response = await unified_search(search_payload)
        search_results = search_response
    
    # Generate narrative
    narrative = await generate_narrative_from_search(search_results, query)
    
    return {
        "query": query,
        "narrative": narrative,
        "search_metadata": search_results.get("metadata", {})
    }


@app.post("/search/dashboard")
async def generate_search_dashboard(payload: Dict[str, Any]) -> Any:
    """
    Generate dashboard configuration from search results.
    
    Request:
    {
        "query": "search query",
        "search_results": {...},  // Optional
        "enable_framework": true,
        "enable_plot": true
    }
    """
    query = payload.get("query", "")
    search_results = payload.get("search_results")
    
    # If search results not provided, perform search first
    if not search_results:
        search_payload = {
            "query": query,
            "enable_framework": True,
            "enable_plot": True
        }
        search_response = await unified_search(search_payload)
        search_results = search_response
    
    # Generate dashboard
    dashboard = await generate_dashboard_from_search(search_results, query)
    
    return {
        "query": query,
        "dashboard": dashboard,
        "search_metadata": search_results.get("metadata", {})
    }


@app.post("/search/narrative-dashboard")
async def generate_narrative_and_dashboard(payload: Dict[str, Any]) -> Any:
    """
    Generate both narrative and dashboard from search results.
    """
    query = payload.get("query", "")
    search_results = payload.get("search_results")
    
    # If search results not provided, perform search first
    if not search_results:
        search_payload = {
            "query": query,
            "enable_framework": True,
            "enable_plot": True
        }
        search_response = await unified_search(search_payload)
        search_results = search_response
    
    # Generate both
    result = await generate_narrative_and_dashboard(search_results, query)
    
    return result
```

## UI Integration

### Enhanced Search Module

```typescript
// Add new buttons/options
<Button
  variant="outlined"
  onClick={handleGenerateNarrative}
  disabled={!hasSearched || loading}
>
  Generate Narrative
</Button>

<Button
  variant="outlined"
  onClick={handleGenerateDashboard}
  disabled={!hasSearched || loading}
>
  Generate Dashboard
</Button>

// New tabs
<Tabs>
  <Tab label="Results" />
  <Tab label="Narrative" />
  <Tab label="Dashboard" />
  <Tab label="Visualization" />
</Tabs>
```

## Benefits

### 1. Dynamic Content Generation
- **No Manual Configuration**: Dashboards and narratives generated automatically
- **Context-Aware**: Content adapts to search results
- **Intelligent**: LLM understands data patterns and generates insights

### 2. User Experience
- **One Query, Multiple Outputs**: Search → Results + Narrative + Dashboard
- **Comprehensive Analysis**: Users get both data and interpretation
- **Visual Understanding**: Dashboards make data patterns clear

### 3. Scalability
- **Works with Any Query**: No need to pre-define dashboard templates
- **Adaptive**: Handles different data types and structures
- **Extensible**: Easy to add new narrative/dashboard types

## Challenges and Solutions

### Challenge 1: LLM Latency
**Problem**: Multiple LLM calls add significant latency

**Solutions**:
- Cache narrative/dashboard configs for similar queries
- Use streaming responses for better UX
- Parallel execution where possible
- Offer "quick" vs "detailed" modes

### Challenge 2: Cost
**Problem**: LLM processing for every search can be expensive

**Solutions**:
- Make narrative/dashboard generation optional
- Cache results for common queries
- Use smaller/faster models for simple queries
- Batch processing for multiple queries

### Challenge 3: Consistency
**Problem**: LLM outputs may vary between runs

**Solutions**:
- Use structured output formats (JSON)
- Validate and normalize outputs
- Provide fallback templates
- Use deterministic prompts

### Challenge 4: Quality Control
**Problem**: LLM may generate incorrect or misleading content

**Solutions**:
- Validate dashboard configs against schema
- Sanitize narrative outputs
- Provide user feedback mechanisms
- Include confidence scores

## Example Use Cases

### Use Case 1: Executive Dashboard
**Query**: "Show me customer data quality metrics"

**Flow**:
1. Search finds customer data quality records
2. Framework generates narrative: "Customer data quality is 87% overall, with address completeness being the main issue..."
3. Framework generates dashboard: Bar chart of quality scores, pie chart of issue types, timeline of quality trends
4. User sees both narrative and interactive dashboard

### Use Case 2: Research Report
**Query**: "Find all documents about regulatory compliance"

**Flow**:
1. Search finds compliance-related documents
2. Framework generates narrative: Executive summary, key findings, detailed analysis
3. Framework generates dashboard: Document timeline, topic distribution, source breakdown
4. User gets comprehensive research report

### Use Case 3: Data Discovery
**Query**: "What data sources are available for financial reporting?"

**Flow**:
1. Search finds financial data sources
2. Framework generates narrative: "Found 15 data sources across 3 systems..."
3. Framework generates dashboard: Source catalog, data lineage graph, quality metrics
4. User understands available data landscape

## Implementation Priority

### Phase 1: Core Narrative Generation (High Priority)
- ✅ Framework chain for narrative generation
- ✅ Gateway endpoint `/search/narrative`
- ✅ UI integration with narrative tab
- **Timeline**: 1-2 days

### Phase 2: Dashboard Configuration (High Priority)
- ✅ Framework chain for dashboard generation
- ✅ Gateway endpoint `/search/dashboard`
- ✅ Dashboard config validation
- ✅ UI dashboard renderer
- **Timeline**: 2-3 days

### Phase 3: Combined Generation (Medium Priority)
- ✅ Combined narrative + dashboard endpoint
- ✅ Parallel execution optimization
- ✅ Caching layer
- **Timeline**: 1 day

### Phase 4: Advanced Features (Low Priority)
- ⏳ Custom narrative templates
- ⏳ Dashboard template library
- ⏳ User feedback and refinement
- ⏳ Export capabilities (PDF, PNG)
- **Timeline**: 3-5 days

## Rating Summary

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Innovation** | ⭐⭐⭐⭐⭐ | Novel approach using search as prompt engine |
| **Feasibility** | ⭐⭐⭐⭐ | Requires framework chain implementation |
| **User Value** | ⭐⭐⭐⭐⭐ | High value - automatic insights and dashboards |
| **Performance** | ⭐⭐⭐ | Latency concerns with multiple LLM calls |
| **Cost** | ⭐⭐⭐ | LLM costs need to be managed |
| **Scalability** | ⭐⭐⭐⭐ | Works with any search query |
| **Overall** | ⭐⭐⭐⭐ | **4.3/5** - Excellent concept with some challenges |

## Recommendations

1. **Start with Narrative Generation**: Simpler to implement, high user value
2. **Make it Optional**: Don't force LLM processing on every search
3. **Add Caching**: Cache narratives/dashboards for common queries
4. **Use Streaming**: Stream LLM responses for better UX
5. **Validate Outputs**: Ensure dashboard configs are valid and safe
6. **Provide Fallbacks**: Have template-based fallbacks if LLM fails

## Conclusion

Using **search as a prompt engine** for dynamic dashboards and narratives is a **highly innovative and valuable approach**. It transforms search from a simple data retrieval mechanism into an intelligent analysis and visualization platform.

**Key Strengths**:
- Automatic content generation
- Context-aware insights
- No manual configuration needed
- Works with any search query

**Key Considerations**:
- Latency management
- Cost optimization
- Quality control
- Output validation

**Recommendation**: **Proceed with implementation**, starting with narrative generation, then dashboard generation, with careful attention to performance and cost optimization.

