# LangGraph Integration for Narrative Intelligence

Natural language interface and orchestration layer for the narrative spacetime GNN system.

## Overview

This module provides a LangGraph-based agent that:
- Routes natural language queries to appropriate GNN capabilities
- Enriches structured GNN outputs with natural language
- Maintains conversational memory and context
- Supports multi-turn narrative reasoning

## Architecture

```
User Query (Natural Language)
    ↓
LangGraph Agent (Query Classification & Context Extraction)
    ↓
MultiPurposeNarrativeGNN (Structured Reasoning)
    ↓
LangGraph Bridge (Response Enrichment)
    ↓
Natural Language Response
```

## Components

### 1. NarrativeLangGraphAgent

Main orchestration agent that routes queries and generates responses.

```python
from langgraph_narrative import NarrativeLangGraphAgent
from gnn_spacetime.narrative import MultiPurposeNarrativeGNN, NarrativeGraph

# Initialize
gnn = MultiPurposeNarrativeGNN(narrative_graph=graph)
agent = NarrativeLangGraphAgent(narrative_gnn=gnn, narrative_graph=graph)

# Chat interface
response = agent.chat("Explain why the merger failed")
```

### 2. LangGraphNarrativeBridge

Converts between natural language and GNN parameters.

```python
from langgraph_narrative import LangGraphNarrativeBridge

bridge = LangGraphNarrativeBridge()

# Convert query to GNN params
params = bridge.langgraph_to_gnn("What happened last year with the merger?")

# Enrich GNN output
enriched = bridge.gnn_to_langgraph(gnn_output, "explain")
```

### 3. NarrativeConversationMemory

Tracks conversation history and extracts narrative elements.

```python
from langgraph_narrative import NarrativeConversationMemory

memory = NarrativeConversationMemory(conversation_graph=graph)

# Update from conversation
memory.update_from_conversation(user_query, gnn_response, llm_response)

# Get context
context = memory.get_conversation_context(num_exchanges=5)
```

## Usage Examples

### Basic Chat

```python
# Initialize system
graph = load_narrative_graph()
gnn = MultiPurposeNarrativeGNN(narrative_graph=graph)
agent = NarrativeLangGraphAgent(narrative_gnn=gnn, narrative_graph=graph)

# Ask questions
queries = [
    "Explain why the merger between Company A and B failed last year",
    "What's likely to happen with this partnership in the next 6 months?",
    "Are there any unusual patterns in this collaboration timeline?",
    "What if the CEO had made different decisions in 2020?"
]

for query in queries:
    response = agent.chat(query, graph)
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

### Multi-Turn Conversation

```python
# With conversation memory
memory = NarrativeConversationMemory(conversation_graph=graph)

# First query
response1 = agent.chat("Explain the merger failure", graph)
memory.update_from_conversation("Explain the merger failure", {}, response1)

# Follow-up query (uses context)
response2 = agent.chat("What about the cultural factors?", graph)
memory.update_from_conversation("What about the cultural factors?", {}, response2)
```

## Query Types Supported

1. **Explain**: "Explain why X happened"
2. **Predict**: "What will happen with Y?"
3. **Detect Anomalies**: "Are there any unusual patterns?"
4. **What-If**: "What if Z had been different?"

## Integration with Graph Service

The LangGraph agent can be integrated with the Go graph service via HTTP:

```python
# In Python service
agent = NarrativeLangGraphAgent(...)

# Expose via FastAPI
@app.post("/narrative/chat")
async def chat(request: ChatRequest):
    response = agent.chat(request.query, graph)
    return {"response": response}
```

## Workflow Structure

The agent uses a LangGraph workflow with these nodes:

1. **classify_query**: Determines query type (explain/predict/detect_anomalies/what_if)
2. **extract_context**: Extracts entities, time periods, themes
3. **execute_gnn**: Runs narrative GNN
4. **generate_response**: Enriches output with LLM

## Next Steps

- [ ] Add HTTP API endpoints
- [ ] Integrate with Go graph service
- [ ] Add streaming responses
- [ ] Implement conversation persistence
- [ ] Add feedback learning from conversations

