# Lang Infrastructure Review

## Executive Summary

This document provides a comprehensive review of all language model and agent infrastructure components in the aModels codebase. The infrastructure consists of five third-party libraries and six services that integrate to provide a complete agentic AI platform.

## Table of Contents

1. [Third-Party Lang Libraries](#third-party-lang-libraries)
2. [Services with Lang Infrastructure](#services-with-lang-infrastructure)
3. [Integration Architecture](#integration-architecture)
4. [Component Inventory](#component-inventory)
5. [API Interfaces](#api-interfaces)
6. [Configuration Reference](#configuration-reference)
7. [Dependencies and Relationships](#dependencies-and-relationships)
8. [Recommendations](#recommendations)

---

## Third-Party Lang Libraries

All third-party libraries are located in `infrastructure/third_party/`.

### 1. LangChain (`infrastructure/third_party/langchain/`)

**Purpose**: Python framework for building LLM-powered applications

**Key Features**:
- Chains, prompts, memory, tools, agents
- Model interoperability
- Real-time data augmentation
- Integration with vector stores, retrievers, and more

**Usage in Codebase**:
- Used as reference library
- Imported by Python services (deepagents, agentflow)
- Provides patterns for other components

**Location**: `infrastructure/third_party/langchain/`

**Documentation**: [LangChain Docs](https://docs.langchain.com/oss/python/langchain/overview)

---

### 2. LangGraph (`infrastructure/third_party/langgraph/`)

**Purpose**: Low-level orchestration framework for building stateful, long-running agents

**Key Features**:
- Graph-based state management
- Durable execution with checkpointing
- Human-in-the-loop workflows
- Comprehensive memory (short-term and long-term)
- Production-ready deployment

**Usage in Codebase**:
- **Primary Integration**: `services/graph/` uses LangGraph Go SDK (`github.com/langchain-ai/langgraph-go`)
- Unified workflow processor (`services/graph/pkg/workflows/unified_processor.go`)
- Workflow orchestration for combining all lang components
- State management and conditional routing

**Location**: `infrastructure/third_party/langgraph/`

**Go SDK**: `github.com/langchain-ai/langgraph-go/pkg/stategraph`

**Documentation**: [LangGraph Docs](https://langchain-ai.github.io/langgraph/)

---

### 3. LangFlow (`infrastructure/third_party/langflow/`)

**Purpose**: Visual workflow builder for creating AI-powered agents and workflows

**Key Features**:
- Visual builder interface
- Interactive playground
- Multi-agent orchestration
- Deploy as API or export as JSON
- MCP server support

**Usage in Codebase**:
- **Primary Integration**: `services/agentflow/` bridges to LangFlow
- Flow execution from JSON files
- Flow catalog management
- Frontend UI for flow creation

**Location**: `infrastructure/third_party/langflow/`

**Documentation**: [LangFlow Docs](https://docs.langflow.org/)

---

### 4. LangExtract (`infrastructure/third_party/langextract/`)

**Purpose**: Structured information extraction from unstructured text using LLMs

**Key Features**:
- Precise source grounding (maps extractions to source text)
- Reliable structured outputs with schema enforcement
- Optimized for long documents (chunking, parallel processing)
- Interactive visualization
- Flexible LLM support (Gemini, OpenAI, Ollama)

**Usage in Codebase**:
- **Primary Integration**: `services/extract/` uses LangExtract via HTTP API
- Entity extraction from documents
- Regulatory extraction (MAS610, BCBS239)
- Audit trail integration (`services/extract/langextract/audit_trail.go`)

**Location**: `infrastructure/third_party/langextract/`

**Integration Point**: `services/extract/extract_logic.go` → `invokeLangextract()`

**Documentation**: [LangExtract GitHub](https://github.com/google/langextract)

---

### 5. Orchestration Framework (`infrastructure/third_party/orchestration/`)

**Purpose**: Go-native LangChain-like framework for LLM applications

**Key Features**:
- LLM chains, prompts, memory, tools, agents
- Go-native implementation
- Supports LocalAI, OpenAI, and other LLM providers
- Document loaders and text splitters
- Vector stores and embeddings

**Components**:
- `llms/`: LLM interfaces and implementations
- `chains/`: Composable chain operations
- `agents/`: Autonomous agents with tools
- `prompts/`: Prompt templates
- `memory/`: Conversation history management
- `tools/`: External tool integrations
- `embeddings/`: Text embedding functionality
- `vectorstores/`: Vector database interfaces

**Usage in Codebase**:
- **Note**: Currently using stubs in `services/graph/pkg/stubs/orchestration.go`
- Framework exists but not directly imported in services
- Orchestration service (`services/orchestration/`) implements agent coordination but doesn't directly use this framework
- Chain creation uses stub implementations that call LocalAI directly

**Location**: `infrastructure/third_party/orchestration/`

**Status**: Framework exists but integration is incomplete (using stubs)

**Documentation**: `infrastructure/third_party/orchestration/README.md`

---

## Services with Lang Infrastructure

### 1. DeepAgents Service (`services/deepagents/`)

**Purpose**: General-purpose tool-using agent service

**Technology**: Python, FastAPI, deepagents library

**Port**: 9004

**Key Features**:
- Planning & task decomposition using todo lists
- Sub-agent spawning for specialized tasks
- File system access tools
- Integration with Knowledge Graph, AgentFlow, and Orchestration

**Tools**:
1. `query_knowledge_graph`: Query Neo4j using Cypher
2. `run_agentflow_flow`: Execute LangFlow flows
3. `run_orchestration_chain`: Execute orchestration chains
4. Built-in deepagents tools: `write_todos`, `task`, `ls`, `read_file`, `write_file`, etc.

**API Endpoints**:
- `GET /healthz`: Health check
- `POST /invoke`: Invoke agent with conversation
- `POST /stream`: Stream agent responses (SSE)
- `GET /agent/info`: Get agent information

**Configuration**:
```bash
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081
ANTHROPIC_API_KEY=your_key  # Preferred
OPENAI_API_KEY=your_key     # Alternative
LOCALAI_URL=http://localai:8080  # Fallback
DEEPAGENTS_PORT=9004
```

**Files**:
- `main.py`: FastAPI application
- `agent_factory.py`: Agent creation
- `tools/`: Tool implementations
  - `knowledge_graph_tool.py`
  - `agentflow_tool.py`
  - `orchestration_tool.py`
  - `gpu_tool.py`
  - `signavio_tool.py`

**Integration Points**:
- Called by extract service for graph analysis
- Used in unified workflow for agent-based analysis
- Can orchestrate all other services via tools

---

### 2. AgentFlow Service (`services/agentflow/`)

**Purpose**: Runs specialized workflows from JSON files, bridges to LangFlow

**Technology**: Go CLI + FastAPI service (Python)

**Port**: 9001

**Key Features**:
- Flow execution from JSON files
- Flow catalog management
- SGMI view lineage integration
- Go CLI for syncing and running flows
- FastAPI HTTP interface

**Components**:
- `cmd/flow-run/`: Go CLI for flow execution
- `service/`: FastAPI service for HTTP interface
- `flows/`: JSON flow definitions
- `frontend/`: React UI for flow management
- `runner/`: Flow execution engine

**API Endpoints**:
- Flow execution endpoints
- Flow catalog management
- Flow synchronization

**Configuration**:
```bash
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
```

**Files**:
- `service/main.py`: FastAPI application
- `service/services/langflow.py`: LangFlow integration
- `runner/runner.go`: Flow execution
- `flows/*.json`: Flow definitions

**Integration Points**:
- Called by graph service unified workflow
- Used by deepagents via `run_agentflow_flow` tool
- Integrates with knowledge graph for data flow

---

### 3. Orchestration Service (`services/orchestration/`)

**Purpose**: Agent coordination and orchestration

**Technology**: Go

**Key Features**:
- Agent coordinator for multi-agent workflows
- Agent factory and marketplace
- Specialized agents:
  - `DataIngestionAgent`: Autonomous data ingestion
  - `MappingRuleAgent`: Schema mapping rules
  - `AnomalyDetectionAgent`: Anomaly detection
  - `TestGenerationAgent`: Test scenario generation
- Digital twin integration
- Auto-pipeline orchestration

**Components**:
- `agent_coordinator.go`: Agent coordination
- `agents/`: Agent implementations
  - `agent_factory.go`: Agent creation
  - `agent_marketplace.go`: Agent discovery
  - `data_ingestion_agent.go`
  - `mapping_rule_agent.go`
  - `anomaly_detection_agent.go`
  - `test_generation_agent.go`
- `digitaltwin/`: Digital twin system
- `api/`: HTTP API handlers

**Agent Types**:
1. **DataIngestionAgent**: Ingests data from source systems (Murex, SAP GL, BCRS, RCO, Axiom)
2. **MappingRuleAgent**: Maps source schemas to knowledge graph schemas
3. **AnomalyDetectionAgent**: Detects anomalies in data
4. **TestGenerationAgent**: Generates and runs test scenarios

**API Endpoints**:
- `api/agents_handler.go`: Agent management endpoints
- `api/digitaltwin_handler.go`: Digital twin endpoints

**Integration Points**:
- Used by extract service for chain matching
- Can be called by graph service (though currently uses stubs)
- Provides agent coordination for complex workflows

**Note**: This service implements agent coordination but doesn't directly use the orchestration framework from `infrastructure/third_party/orchestration/`. Instead, it provides a higher-level agent management system.

---

### 4. Extract Service (`services/extract/`)

**Purpose**: Structured data extraction using LangExtract

**Technology**: Go

**Port**: 19080

**Key Features**:
- Entity extraction from documents using LangExtract
- Regulatory extraction (MAS610, BCBS239)
- Knowledge graph persistence
- Integration with orchestration chains
- Integration with deepagents

**Integration Points**:

1. **LangExtract Integration**:
   - `extract_logic.go`: Main extraction logic
   - `invokeLangextract()`: Calls LangExtract HTTP API
   - `langextract/audit_trail.go`: Audit trail for extractions

2. **Orchestration Integration**:
   - `orchestration_integration.go`: `OrchestrationChainMatcher`
   - Semantic chain matching for task routing
   - Classification-based routing

3. **DeepAgents Integration**:
   - `deepagents.go`: `DeepAgentsClient`
   - Graph analysis and insights
   - Non-fatal integration (graceful degradation)

**Configuration**:
```bash
LANGEXTRACT_URL=http://langextract-service:port
LOCALAI_URL=http://localai:8080
DEEPAGENTS_URL=http://deepagents-service:9004
DEEPAGENTS_ENABLED=true  # Default: enabled
```

**Files**:
- `extract_logic.go`: Core extraction logic
- `orchestration_integration.go`: Orchestration chain matching
- `deepagents.go`: DeepAgents client
- `langextract/`: LangExtract integration
- `regulatory/`: Regulatory extraction (MAS610, BCBS239)

---

### 5. Graph Service (`services/graph/`)

**Purpose**: LangGraph-based unified workflow processor

**Technology**: Go, LangGraph Go SDK

**Port**: 8081

**Key Features**:
- Unified workflow that combines all lang infrastructure
- LangGraph-based state management
- Workflow modes: sequential, parallel, conditional
- GPU orchestration integration
- GraphRAG queries

**Components**:

1. **Unified Workflow** (`pkg/workflows/unified_processor.go`):
   - Entry point for all lang components
   - Combines Knowledge Graph, Orchestration, AgentFlow, DeepAgents
   - Supports sequential, parallel, and conditional execution

2. **Workflow Processors**:
   - `knowledge_graph_processor.go`: Knowledge graph processing
   - `orchestration_processor.go`: Orchestration chain execution
   - `agentflow_processor.go`: AgentFlow flow execution
   - `deepagents_processor.go`: DeepAgents integration
   - `graphrag_processor.go`: GraphRAG queries
   - `gpu_processor.go`: GPU allocation

3. **Orchestration Integration** (`pkg/stubs/orchestration.go`):
   - Stub implementations for orchestration chains
   - Chain creation based on chain name
   - Supports: llm_chain, question_answering, summarization, knowledge_graph_analyzer, data_quality_analyzer, pipeline_analyzer, sql_analyzer, agentflow_analyzer

**Workflow Modes**:
- **Sequential**: KG → Orchestration → AgentFlow
- **Parallel**: All components execute simultaneously
- **Conditional**: Route based on results/quality

**API Endpoints**:
- Unified workflow endpoint
- Individual processor endpoints
- GraphRAG query endpoint

**Configuration**:
```bash
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
LOCALAI_URL=http://localai:8080
GPU_ORCHESTRATOR_URL=http://gpu-orchestrator:port
```

**Files**:
- `pkg/workflows/unified_processor.go`: Main unified workflow
- `pkg/workflows/orchestration_processor.go`: Orchestration integration
- `pkg/stubs/orchestration.go`: Chain stubs
- `cmd/graph-server/main.go`: Server entry point

---

### 6. LocalAI Service (`services/localai/`)

**Purpose**: Local LLM inference backend

**Technology**: Go, multiple backends

**Port**: 8080

**Key Features**:
- Local LLM inference
- Multiple model backends
- LangChain integration (`pkg/langchain/langchain.go`)
- gRPC-based architecture
- Supports text generation, embeddings, image generation, audio processing

**Components**:
- Backend implementations for different languages
- LangChain compatibility layer
- Model management
- API server

**Integration Points**:
- Used by all services that need LLM inference
- Orchestration chains call LocalAI
- DeepAgents can use LocalAI as fallback
- Extract service uses LocalAI for domain detection

**Configuration**:
```bash
LOCALAI_URL=http://localai:8080
```

---

## Integration Architecture

### Unified Workflow Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Graph Service                            │
│              (Unified Workflow Processor)                   │
│                                                             │
│  Entry → GPU Allocation → [Workflow Mode Selection]       │
│                                                             │
│  Sequential Mode:                                           │
│    KG Processing → Orchestration → AgentFlow → DeepAgents  │
│                                                             │
│  Parallel Mode:                                             │
│    [KG, Orchestration, AgentFlow, DeepAgents] → Join        │
│                                                             │
│  Conditional Mode:                                          │
│    Route based on quality/results                          │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Integration Mapping

#### 1. Extract → Orchestration Integration

**File**: `services/extract/orchestration_integration.go`

**Component**: `OrchestrationChainMatcher`

**Purpose**: Routes extraction tasks to appropriate orchestration chains

**Mechanism**:
- Classification-based routing (transaction, reference, staging, test)
- Semantic search-based matching (optional, via `USE_SAP_RPT_EMBEDDINGS`)
- Fallback to default chain

**Methods**:
- `MatchChainToTask()`: Matches chain to task using classification or semantic search
- `SelectChainForTable()`: Selects chain for a table based on knowledge graph classification

**Chain Types**:
- `transaction_processing_chain`: For transaction tables
- `reference_lookup_chain`: For reference data
- `staging_etl_chain`: For staging tables
- `test_processing_chain`: For test data
- `default_chain`: Fallback

#### 2. Extract → DeepAgents Integration

**File**: `services/extract/deepagents.go`

**Component**: `DeepAgentsClient`

**Purpose**: Provides AI-powered graph analysis and insights

**Mechanism**:
- HTTP client to DeepAgents service
- Non-fatal integration (returns `nil, nil` on failure)
- Health check before attempting analysis
- Retry logic with exponential backoff

**Methods**:
- `AnalyzeKnowledgeGraph()`: Analyzes knowledge graph structure and quality
- `FormatGraphSummary()`: Formats graph data for analysis

**Features**:
- Graceful degradation (continues if service unavailable)
- Health check timeout: 5 seconds
- Max retries: 2
- Request timeout: 120 seconds

#### 3. Graph → Orchestration Integration

**File**: `services/graph/pkg/workflows/orchestration_processor.go`

**Component**: `RunOrchestrationChainNode`

**Purpose**: Executes orchestration chains with knowledge graph context

**Mechanism**:
- Creates chains via `createOrchestrationChain()` (uses stubs)
- Enriches chain inputs with KG context (quality metrics, node/edge counts, query results)
- Executes chain via LocalAI
- Analyzes results and routes conditionally

**Chain Types Supported**:
- `llm_chain`, `default`: Basic LLM chain
- `question_answering`, `qa`: Context-aware Q&A
- `summarization`, `summarize`: Text summarization
- `knowledge_graph_analyzer`, `kg_analyzer`: KG analysis
- `data_quality_analyzer`, `quality_analyzer`: Data quality analysis
- `pipeline_analyzer`, `pipeline`: Pipeline analysis
- `sql_analyzer`, `sql`: SQL analysis
- `agentflow_analyzer`, `agentflow`: AgentFlow analysis

**Context Enrichment**:
- Quality metrics: `quality_score`, `quality_level`, `issues`
- Graph structure: `node_count`, `edge_count`
- Query results: `knowledge_graph_query_results`
- Information theory: `metadata_entropy`, `kl_divergence`

#### 4. Graph → AgentFlow Integration

**File**: `services/graph/pkg/workflows/agentflow_processor.go`

**Component**: `RunAgentFlowFlowNode`

**Purpose**: Executes AgentFlow flows with KG and orchestration data

**Mechanism**:
- Calls AgentFlow service HTTP API
- Passes knowledge graph context
- Passes orchestration results as inputs
- Handles flow execution and result processing

**Data Flow**:
```
KG Results → AgentFlow Inputs
Orchestration Text → AgentFlow Inputs
```

#### 5. Graph → DeepAgents Integration

**File**: `services/graph/pkg/workflows/deepagents_processor.go`

**Component**: `RunDeepAgentNode`

**Purpose**: Executes deep agent analysis

**Mechanism**:
- Calls DeepAgents service HTTP API
- Passes context from previous workflow steps
- Processes agent responses and tool calls

#### 6. DeepAgents → All Services Integration

**Files**: `services/deepagents/tools/*.py`

**Tools**:

1. **Knowledge Graph Tool** (`knowledge_graph_tool.py`):
   - Calls: `EXTRACT_SERVICE_URL/knowledge-graph/query`
   - Purpose: Query Neo4j knowledge graph
   - Input: Cypher query, optional project_id, system_id
   - Output: Formatted query results

2. **AgentFlow Tool** (`agentflow_tool.py`):
   - Calls: `AGENTFLOW_SERVICE_URL/run`
   - Purpose: Execute LangFlow flows
   - Input: Flow ID, inputs
   - Output: Flow execution results

3. **Orchestration Tool** (`orchestration_tool.py`):
   - Calls: `GRAPH_SERVICE_URL/orchestration/process`
   - Purpose: Execute orchestration chains
   - Input: Chain name, inputs, optional KG query
   - Output: Chain execution results

4. **GPU Tool** (`gpu_tool.py`):
   - Calls: GPU orchestrator service
   - Purpose: Allocate/release GPU resources

5. **Signavio Tool** (`signavio_tool.py`):
   - Purpose: Signavio process intelligence integration

**Integration Pattern**:
- All tools use `@tool` decorator from `langchain_core.tools`
- HTTP clients with timeout configuration
- Error handling and formatted responses
- Tool descriptions guide LLM usage

### Integration Points Summary

1. **Extract → Orchestration**:
   - `OrchestrationChainMatcher` routes tasks to chains
   - Semantic matching and classification-based routing
   - File: `services/extract/orchestration_integration.go`

2. **Extract → DeepAgents**:
   - `DeepAgentsClient` for graph analysis
   - Non-fatal integration (graceful degradation)
   - File: `services/extract/deepagents.go`

3. **Graph → Orchestration**:
   - Chain execution with KG context enrichment
   - Stub-based implementation (calls LocalAI directly)
   - File: `services/graph/pkg/workflows/orchestration_processor.go`

4. **Graph → AgentFlow**:
   - Flow execution with KG data
   - File: `services/graph/pkg/workflows/agentflow_processor.go`

5. **Graph → DeepAgents**:
   - Agent-based analysis
   - File: `services/graph/pkg/workflows/deepagents_processor.go`

6. **DeepAgents → All**:
   - Tool-based access to all services
   - `query_knowledge_graph`, `run_agentflow_flow`, `run_orchestration_chain`
   - Files: `services/deepagents/tools/*.py`

### Data Flow Example

**Knowledge Graph Analysis Workflow**:

```
1. Extract Service extracts data → Knowledge Graph
2. Graph Service processes KG → Enriches with quality metrics
3. Orchestration Chain analyzes KG → Generates insights
4. AgentFlow flow processes insights → Creates recommendations
5. DeepAgents coordinates → Final analysis and report
```

---

## Component Inventory

### Third-Party Libraries

| Component | Location | Language | Purpose | Status |
|-----------|----------|----------|---------|--------|
| LangChain | `infrastructure/third_party/langchain/` | Python | LLM framework | Reference |
| LangGraph | `infrastructure/third_party/langgraph/` | Python/Go | Workflow orchestration | Active (Go SDK) |
| LangFlow | `infrastructure/third_party/langflow/` | Python | Visual workflow builder | Active |
| LangExtract | `infrastructure/third_party/langextract/` | Python | Structured extraction | Active |
| Orchestration | `infrastructure/third_party/orchestration/` | Go | LangChain-like framework | Exists (stubs used) |

### Services

| Service | Location | Port | Language | Purpose | Status |
|---------|----------|------|----------|---------|--------|
| DeepAgents | `services/deepagents/` | 9004 | Python | Tool-using agent | Active |
| AgentFlow | `services/agentflow/` | 9001 | Go/Python | Flow execution | Active |
| Orchestration | `services/orchestration/` | - | Go | Agent coordination | Active |
| Extract | `services/extract/` | 19080 | Go | Data extraction | Active |
| Graph | `services/graph/` | 8081 | Go | Unified workflow | Active |
| LocalAI | `services/localai/` | 8080 | Go | LLM backend | Active |

---

## API Interfaces

### DeepAgents Service

**Base URL**: `http://deepagents-service:9004`

**Endpoints**:
- `GET /healthz`: Health check
- `POST /invoke`: Invoke agent
  ```json
  {
    "messages": [{"role": "user", "content": "..."}],
    "stream": false,
    "config": {}
  }
  ```
- `POST /stream`: Stream responses (SSE)
- `GET /agent/info`: Agent information

### AgentFlow Service

**Base URL**: `http://agentflow-service:9001`

**Endpoints**:
- Flow execution endpoints
- Flow catalog management
- Flow synchronization

### Extract Service

**Base URL**: `http://extract-service:19080`

**Endpoints**:
- `POST /extract`: Extract entities
- `POST /knowledge-graph/query`: Query knowledge graph
- `POST /knowledge-graph/search`: Semantic search

### Graph Service

**Base URL**: `http://graph-service:8081`

**Endpoints**:
- `POST /unified/process`: Unified workflow
- `POST /orchestration/process`: Orchestration chain
- `POST /agentflow/run`: AgentFlow flow
- `POST /graphrag/query`: GraphRAG query

### LocalAI Service

**Base URL**: `http://localai:8080`

**Endpoints**:
- Standard LocalAI API endpoints
- LangChain-compatible interface

---

## Configuration Reference

### Environment Variables

#### DeepAgents
```bash
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
LOCALAI_URL=http://localai:8080
DEEPAGENTS_PORT=9004
DEEPAGENTS_ENABLED=true  # Default: enabled
```

#### AgentFlow
```bash
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
```

#### Extract
```bash
LANGEXTRACT_URL=http://langextract-service:port
LOCALAI_URL=http://localai:8080
DEEPAGENTS_URL=http://deepagents-service:9004
DEEPAGENTS_ENABLED=true  # Default: enabled
EXTRACT_SERVICE_URL=http://extract-service:19080
```

#### Graph
```bash
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
LOCALAI_URL=http://localai:8080
GPU_ORCHESTRATOR_URL=http://gpu-orchestrator:port
```

#### LocalAI
```bash
LOCALAI_URL=http://localai:8080
```

---

## Dependencies and Relationships

### Dependency Graph

```
┌─────────────┐
│  LangChain  │ (Reference)
└─────────────┘

┌─────────────┐
│  LangGraph  │──┐
└─────────────┘  │
                 ├──→ Graph Service (Go SDK)
┌─────────────┐  │
│  LangFlow   │──┘
└─────────────┘
      │
      └──→ AgentFlow Service

┌─────────────┐
│ LangExtract │──→ Extract Service
└─────────────┘

┌─────────────┐
│Orchestration│ (Framework exists but uses stubs)
└─────────────┘
      │
      └──→ Graph Service (via stubs)

┌─────────────┐
│  LocalAI    │──→ All Services (LLM backend)
└─────────────┘
```

### Service Dependencies

**Graph Service** depends on:
- Extract Service (KG processing)
- AgentFlow Service (flow execution)
- LocalAI (LLM inference)
- GPU Orchestrator (optional)

**Extract Service** depends on:
- LangExtract (extraction)
- LocalAI (domain detection)
- DeepAgents (analysis, optional)
- Orchestration (chain matching)

**DeepAgents** depends on:
- Extract Service (KG queries)
- AgentFlow Service (flow execution)
- Graph Service (KG access)
- LocalAI/Anthropic/OpenAI (LLM)

**AgentFlow** depends on:
- LangFlow (flow execution)

---

## Recommendations

### 1. Complete Orchestration Framework Integration

**Current State**: The orchestration framework exists in `infrastructure/third_party/orchestration/` but services use stubs (`services/graph/pkg/stubs/orchestration.go`).

**Recommendation**: 
- Replace stubs with actual orchestration framework imports
- Integrate framework into graph service orchestration processor
- Use framework's chain implementations instead of direct LocalAI calls

**Impact**: Better chain management, more features, proper abstraction

### 2. Consolidate Agent Definitions

**Current State**: Agents are defined in multiple places:
- `services/orchestration/agents/`: Agent coordination
- `services/deepagents/`: Tool-using agent
- Stub implementations in graph service

**Recommendation**:
- Create a unified agent registry
- Standardize agent interfaces
- Document agent capabilities and usage

### 3. Improve Documentation

**Current State**: Documentation exists but is scattered.

**Recommendation**:
- Create a central lang infrastructure documentation hub
- Document all integration points
- Add architecture diagrams
- Create usage examples for each component

### 4. Standardize Configuration

**Current State**: Configuration is spread across multiple services.

**Recommendation**:
- Create a unified configuration file
- Document all environment variables
- Provide configuration validation
- Use configuration management tool

### 5. Enhance Error Handling

**Current State**: Some integrations use graceful degradation (e.g., DeepAgents in extract service).

**Recommendation**:
- Standardize error handling patterns
- Implement retry logic consistently
- Add circuit breakers for external services
- Improve error messages and logging

### 6. Testing and Validation

**Recommendation**:
- Add integration tests for all workflows
- Test each integration point
- Validate configuration
- Performance testing for unified workflow

### 7. Code Organization

**Current State**: Lang infrastructure is spread across services and third_party.

**Recommendation**:
- Consider creating a top-level `agents/` directory (as originally planned)
- Group related components together
- Improve discoverability of lang infrastructure
- Document component relationships

### 8. Monitoring and Observability

**Recommendation**:
- Add metrics for all lang components
- Track chain execution times
- Monitor agent performance
- Add distributed tracing

---

## Conclusion

The aModels codebase has a comprehensive lang infrastructure with five third-party libraries and six services. The infrastructure is functional but has opportunities for improvement:

1. **Strengths**:
   - Comprehensive coverage of lang infrastructure needs
   - Good integration between components
   - Flexible workflow modes (sequential, parallel, conditional)
   - Graceful degradation where appropriate

2. **Areas for Improvement**:
   - Complete orchestration framework integration
   - Better code organization
   - Enhanced documentation
   - Standardized configuration
   - Improved testing

3. **Next Steps**:
   - Prioritize orchestration framework integration
   - Create unified documentation
   - Standardize configuration management
   - Add comprehensive testing

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-27  
**Author**: Lang Infrastructure Review

