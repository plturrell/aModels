# DeepAgents Integration Proposal

## Overview

[`deepagents`](https://pypi.org/project/deepagents/) is a Python package built on LangGraph that provides "deep agent" capabilities with planning, sub-agent spawning, file system access, and todo list management. This document proposes integrating `deepagents` into `aModels` to enhance agent capabilities.

## Why DeepAgents?

### Current State

`aModels` currently has:
- ✅ **Go-native Orchestration**: LangChain-like framework in Go
- ✅ **LangGraph Workflows**: Stateful workflows in Go
- ✅ **AgentFlow**: LangFlow-based visual flow builder
- ✅ **Knowledge Graphs**: Neo4j integration for context
- ❌ **Planning Capabilities**: No built-in task decomposition
- ❌ **Sub-Agent Spawning**: Limited agent hierarchy
- ❌ **File System Context**: No persistent file system for agents
- ❌ **Todo Management**: No planning and task tracking

### DeepAgents Capabilities

`deepagents` provides:
1. **Planning & Task Decomposition**: `write_todos` tool for breaking down complex tasks
2. **Sub-Agent Spawning**: `task` tool for spawning specialized subagents
3. **File System Tools**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` for context management
4. **Long-term Memory**: LangGraph Store integration for persistent memory
5. **Built-in System Prompt**: Detailed prompt for deep agent behavior

## Integration Architecture

### Option 1: Python Service (Recommended)

Create a new `deepagents` service that wraps `deepagents` and integrates with existing services.

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

### Service Structure

```
services/deepagents/
├── Dockerfile
├── requirements.txt          # deepagents, langchain, etc.
├── main.py                   # FastAPI service
├── agent_factory.py          # Creates deep agents with custom tools
├── tools/
│   ├── knowledge_graph_tool.py   # Tool to query Neo4j
│   ├── agentflow_tool.py          # Tool to run AgentFlow flows
│   └── orchestration_tool.py      # Tool to run orchestration chains
└── README.md
```

### Integration Points

1. **Gateway Integration**
   - Add `/deepagents/invoke` endpoint
   - Proxy to deepagents service
   - Health check endpoint

2. **Knowledge Graph Integration**
   - Create `knowledge_graph_tool` that queries Neo4j
   - Provides context to deep agents
   - Enables agents to query data lineage, schemas, etc.

3. **AgentFlow Integration**
   - Create `agentflow_tool` that runs LangFlow flows
   - Deep agents can trigger AgentFlow workflows
   - Results feed back into agent context

4. **Orchestration Integration**
   - Create `orchestration_tool` that runs orchestration chains
   - Deep agents can use orchestration for specific tasks
   - Chain results provide context

5. **Unified Workflow Integration**
   - Add deep agent node to LangGraph workflows
   - Deep agents can be part of larger workflows
   - Sub-agents for complex tasks

## Implementation Plan

### Phase 1: Basic Service Setup

1. Create `services/deepagents/` directory
2. Add `Dockerfile` and `requirements.txt`
3. Implement basic FastAPI service with `/health` endpoint
4. Add to `docker/compose.yml`

### Phase 2: Core Deep Agent

1. Implement `agent_factory.py` with `create_deep_agent`
2. Add system prompt for data pipeline analysis
3. Create basic tools (knowledge graph, agentflow, orchestration)
4. Test with simple queries

### Phase 3: Custom Tools Integration

1. **Knowledge Graph Tool**
   ```python
   @tool
   def query_knowledge_graph(
       query: str,
       project_id: str = None,
       system_id: str = None
   ) -> str:
       """Query the Neo4j knowledge graph using Cypher."""
       # Call extract service /knowledge-graph/query
       pass
   ```

2. **AgentFlow Tool**
   ```python
   @tool
   def run_agentflow_flow(
       flow_id: str,
       inputs: dict = None
   ) -> str:
       """Run an AgentFlow/LangFlow flow."""
       # Call agentflow service /flows/{flow_id}/run
       pass
   ```

3. **Orchestration Tool**
   ```python
   @tool
   def run_orchestration_chain(
       chain_name: str,
       inputs: dict
   ) -> str:
       """Run an orchestration chain."""
       # Call graph service /orchestration/process
       pass
   ```

### Phase 4: LangGraph Integration

1. Add `DeepAgentNode` to workflows
2. Integrate with unified workflow
3. Support sub-agent spawning for complex tasks
4. Add conditional routing based on agent results

### Phase 5: Advanced Features

1. **Planning Integration**
   - Use todo lists for complex data pipeline tasks
   - Track progress through multi-step workflows
   - Adapt plans based on results

2. **File System Integration**
   - Use file system tools for context management
   - Store intermediate results
   - Manage large context windows

3. **Memory Integration**
   - Use LangGraph Store for persistent memory
   - Remember previous queries and results
   - Learn from past interactions

## Example Use Cases

### 1. Data Pipeline Analysis

```python
# Deep agent analyzes a Control-M → SQL → Tables pipeline
agent = create_deep_agent(
    tools=[
        query_knowledge_graph,
        run_agentflow_flow,
        run_orchestration_chain,
    ],
    system_prompt="""You are an expert data pipeline analyst.
    You can query knowledge graphs, run AgentFlow flows, and use orchestration chains.
    Break down complex pipeline analysis into steps using todos.
    Use sub-agents for specialized tasks like SQL optimization or data quality analysis.
    """,
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Analyze the SGMI pipeline and suggest optimizations"
    }]
})
```

### 2. Multi-Step Data Quality Assessment

```python
# Deep agent performs comprehensive data quality assessment
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": """Assess data quality for the SGMI system:
        1. Query knowledge graph for all tables
        2. Analyze data quality metrics for each table
        3. Generate recommendations
        4. Create AgentFlow flow for automated quality checks
        """
    }]
})
```

### 3. Automated Pipeline Generation

```python
# Deep agent creates AgentFlow flows from knowledge graph
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": """Create an AgentFlow flow for the Control-M pipeline:
        1. Query knowledge graph for Control-M jobs and SQL
        2. Analyze dependencies
        3. Generate LangFlow flow JSON
        4. Create flow in AgentFlow service
        """
    }]
})
```

## Benefits

1. **Enhanced Planning**: Deep agents can break down complex tasks into manageable steps
2. **Sub-Agent Isolation**: Specialized sub-agents for focused tasks
3. **Context Management**: File system tools prevent context overflow
4. **Long-term Memory**: Persistent memory across conversations
5. **Unified Interface**: Single entry point for complex agent tasks
6. **Python Ecosystem**: Access to Python LLM tools and libraries

## Considerations

### Pros
- ✅ Built on LangGraph (consistent with our architecture)
- ✅ Provides planning and sub-agent capabilities we lack
- ✅ Python ecosystem integration
- ✅ Well-documented and maintained by LangChain team
- ✅ Complements our Go-based orchestration

### Cons
- ⚠️ Adds Python service (we're primarily Go)
- ⚠️ Requires integration with existing services
- ⚠️ Additional service to maintain
- ⚠️ May overlap with some orchestration capabilities

## Recommendation

**Yes, integrate `deepagents`** because:
1. It provides capabilities we don't have (planning, sub-agents, file system)
2. It complements our existing Go-based orchestration
3. It enhances our agent capabilities significantly
4. It's built on LangGraph, which we already use
5. It provides a Python-based agent layer that can leverage Python LLM tools

## Implementation Priority

1. **High**: Basic service setup and core deep agent
2. **High**: Knowledge graph tool integration
3. **Medium**: AgentFlow and orchestration tools
4. **Medium**: LangGraph workflow integration
5. **Low**: Advanced features (planning, memory, file system)

## Next Steps

1. Review and approve this proposal
2. Create `services/deepagents/` directory structure
3. Implement basic service with Docker setup
4. Add gateway integration
5. Implement core tools (knowledge graph, agentflow, orchestration)
6. Test with example use cases
7. Integrate with LangGraph workflows

