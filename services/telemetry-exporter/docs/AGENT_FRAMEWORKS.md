# Agent Framework Instrumentation

This document describes the OpenTelemetry instrumentation for each agent framework in the system.

## Supported Frameworks

### LangGraph
- **Service**: `graph-service`
- **Framework Type**: `langgraph`
- **Instrumentation Points**:
  - Graph execution (`langgraph.execution`)
  - Orchestration chain nodes (`orchestration.chain.{chain_name}`)
  - DeepAgents processor nodes (`deepagents.execute`)
  - Workflow state transitions

**Key Attributes**:
- `agent.framework.type`: "langgraph"
- `workflow.type`: "orchestration" | "deepagents" | "workflow"
- `chain.name`: Orchestration chain identifier
- `workflow.name`: Workflow identifier
- `workflow.version`: Workflow version

### DeepAgents
- **Service**: `deepagents-service` (Python)
- **Framework Type**: `deepagents`
- **Instrumentation Points**:
  - Agent invocation (`deepagents.invoke`)
  - Structured agent invocation (`deepagents.invoke.structured`)
  - Tool usage events
  - LLM call events

**Key Attributes**:
- `agent.framework.type`: "deepagents"
- `workflow.type`: "agent_invocation" | "structured_agent_invocation"
- `request.message_count`: Number of messages in request
- `response.message_count`: Number of messages in response
- `estimated_tokens`: Estimated token usage
- `structured_output.success`: Whether structured output was successful

### Goose
- **Service**: `regulatory-service`
- **Framework Type**: `goose`
- **Instrumentation Points**:
  - Model adapter queries (`model.adapter.goose.query`)
  - Task execution
  - Autonomous agent operations

**Key Attributes**:
- `agent.framework.type`: "goose"
- `model.type`: "goose"
- `query.type`: Query type (compliance, lineage, etc.)
- `principle.id`: BCBS 239 principle ID

### Deep Research
- **Service**: `regulatory-service`
- **Framework Type**: `deep-research`
- **Instrumentation Points**:
  - Model adapter queries (`model.adapter.deepresearch.query`)
  - Research task execution
  - Multi-step analysis

**Key Attributes**:
- `agent.framework.type`: "deep-research"
- `model.type`: "deepresearch"
- `query.type`: Research query type
- `principle.id`: BCBS 239 principle ID

### Browser Automation
- **Service**: `browser-service` (Python)
- **Framework Type**: `browser-automation`
- **Instrumentation Points**:
  - Page navigation (`browser.navigate`)
  - Content extraction (`browser.extract`)
  - Screenshot capture

**Key Attributes**:
- `agent.framework.type`: "browser-automation"
- `browser.url`: Target URL
- `browser.selector`: CSS selector (if used)
- `extracted.text_length`: Length of extracted text
- `extracted.html_length`: Length of extracted HTML

### Regulatory Compliance
- **Service**: `regulatory-service`
- **Framework Type**: `regulatory`
- **Instrumentation Points**:
  - Compliance workflow execution (`compliance.workflow.execute`)
  - Workflow nodes (`compliance.workflow.node.{node_name}`)
  - Model orchestration

**Key Attributes**:
- `agent.framework.type`: "regulatory"
- `workflow.type`: "compliance_reasoning"
- `principle.id`: BCBS 239 principle ID
- `node.name`: Workflow node name
- `confidence`: Confidence score
- `sources.count`: Number of sources

## Common Attributes

All agent frameworks include these standard attributes:

- `service.name`: Service identifier
- `service.instance.id`: Service instance identifier
- `deployment.environment`: Deployment environment
- `agent.framework.type`: Framework type (see above)
- `workflow.type`: Type of workflow/operation
- `latency_ms`: Execution latency in milliseconds

## Events

Common events emitted across frameworks:

- `agent.invoke.completed`: Agent invocation completed
- `agent.invoke.structured.completed`: Structured agent invocation completed
- `workflow.node.completed`: Workflow node execution completed
- `workflow.completed`: Entire workflow completed
- `tool.call`: Tool invocation started
- `tool.result`: Tool invocation completed
- `llm.call`: LLM API call started
- `llm.response`: LLM API response received
- `process.step`: Process step executed
- `workflow.step`: Workflow step executed

## Best Practices

1. **Always Set Framework Type**: Include `agent.framework.type` attribute for proper categorization
2. **Use Descriptive Span Names**: Include context in span names (e.g., `orchestration.chain.extract_data`)
3. **Record Workflow Context**: Include `workflow.name` and `workflow.version` for tracking
4. **Emit Tool Events**: Use `tool.call` and `tool.result` events for tool usage tracking
5. **Track LLM Calls**: Emit `llm.call` and `llm.response` events with token counts
6. **Include Error Context**: Always record errors with `span.record_exception()` and set error status
7. **Add Performance Metrics**: Include latency, token counts, and other performance metrics as attributes

