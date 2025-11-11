# Architecture Overview: Graph-LocalAI-Models Abstraction

## Introduction

This document provides a high-level overview of the three-layer architecture that connects the Graph service (workflow orchestration), LocalAI service (model routing and inference), and the underlying model files. The architecture is designed with clear abstraction boundaries to enable modularity, testability, and flexibility.

## System Layers

### Layer 1: Graph Service (Workflow Orchestration)

**Location**: `services/graph/`

The Graph service uses LangGraph to orchestrate complex workflows. It provides high-level abstractions for:
- Knowledge graph processing
- Orchestration chain execution
- AgentFlow integration
- Unified workflow management

**Key Components**:
- **LangGraph StateGraph**: Core workflow execution engine
- **Orchestration Processor**: `pkg/workflows/orchestration_processor.go`
  - Creates and executes LLM chains
  - Integrates with LocalAI via HTTP client
- **Unified Processor**: `pkg/workflows/unified_processor.go`
  - Combines multiple workflow types
  - Manages sequential and parallel execution

**Abstraction Boundary**: The Graph service communicates with LocalAI through a standard HTTP REST API (OpenAI-compatible), treating LocalAI as a black-box LLM provider.

### Layer 2: LocalAI Service (Domain Routing & Model Management)

**Location**: `services/localai/`

The LocalAI service acts as an intelligent router and model manager. It provides:
- Domain-based routing (24+ specialized agent domains)
- Model loading and caching
- Multiple backend support (SafeTensors, GGUF, Transformers, OCR)
- OpenAI-compatible API interface

**Key Components**:
- **HTTP Server**: `pkg/server/vaultgemma_server.go`
  - Handles `/v1/chat/completions` endpoint
  - Processes requests and routes to appropriate domains
- **Domain Manager**: `pkg/domain/domain_config.go`
  - Loads domain configurations from `config/domains.json`
  - Detects appropriate domain based on prompt keywords
  - Manages domain-specific settings (temperature, max_tokens, etc.)
- **Model Provider Interface**: `pkg/server/interfaces.go`
  - Abstract interface for accessing models
  - Supports multiple model types (SafeTensors, GGUF, Transformers)
- **Request Processor**: Handles the full request lifecycle
  - Domain detection
  - Model resolution
  - Inference execution
  - Response formatting

**Abstraction Boundary**: LocalAI abstracts away model format details and provides a unified interface for inference, regardless of whether models are loaded from SafeTensors, GGUF files, or external Transformers services.

### Layer 3: Models Layer (Physical Model Files)

**Location**: `models/` directory

The models layer contains the actual model files and loaders:
- **Model Formats**:
  - SafeTensors: `*.safetensors` files with `config.json`
  - GGUF: Quantized models for llama.cpp
  - Transformers: External Python service for HuggingFace models
- **Model Loader**: `pkg/models/model_loader.go`
  - Loads models from various formats
  - Validates model structure
  - Provides model metadata
- **Model Registry**: `pkg/server/model_registry.go`
  - Tracks GPU requirements
  - Manages model metadata
  - Estimates model sizes

## Data Flow

```
Graph Workflow
    ↓ (HTTP POST /v1/chat/completions)
LocalAI Server
    ↓ (Domain Detection)
Domain Manager
    ↓ (Model Selection)
Model Provider
    ↓ (Load if needed)
Model Loader
    ↓ (Read from filesystem)
Physical Model Files
```

## Key Interfaces

### 1. Graph → LocalAI Interface

**Protocol**: HTTP REST (OpenAI-compatible)
**Endpoint**: `/v1/chat/completions`
**Client**: `infrastructure/third_party/orchestration/llms/localai/localai.go`

```go
// Graph service creates LocalAI client
llm, err := localai.New(
    localai.WithBaseURL("http://localai:8080"),
    localai.WithHeaders(workflowHeaders),
)

// Makes HTTP request
POST /v1/chat/completions
{
    "model": "auto",
    "messages": [...],
    "temperature": 0.7,
    "max_tokens": 500
}
```

### 2. LocalAI Internal Interfaces

**ModelProvider Interface** (`pkg/server/interfaces.go:17`):
```go
type ModelProvider interface {
    GetSafetensorsModel(key string) (*ai.VaultGemma, bool)
    GetGGUFModel(key string) (*gguf.Model, bool)
    GetTransformerClient(key string) (*transformers.Client, bool)
    HasModel(key string) bool
}
```

**BackendProvider Interface** (`pkg/server/interfaces.go:29`):
```go
type BackendProvider interface {
    GetBackendType(domain string) string
    IsBackendAvailable(backendType string) bool
    GetBackendEndpoint(backendType, domain string) string
}
```

## Configuration Points

### Graph Service Configuration

**Environment Variables**:
- `LOCALAI_URL`: URL of LocalAI service (default: `http://localai:8080`)
- `EXTRACT_SERVICE_URL`: Knowledge graph service URL
- `AGENTFLOW_SERVICE_URL`: AgentFlow service URL

**Code Location**: `services/graph/cmd/graph-server/main.go:288`

### LocalAI Service Configuration

**Domain Configuration**: `config/domains.json`
- Defines available domains
- Maps domains to model paths
- Configures backend types
- Sets domain-specific parameters

**Model Paths**: Configured per domain in `domains.json`
- SafeTensors: `"model_path": "../../models/vaultgemma-1b-transformers"`
- GGUF: `"model_path": "../../models/phi-3.5-mini-instruct.gguf"`
- Transformers: `"transformers_config": {"endpoint": "...", "model_name": "..."}`

**Code Location**: `services/localai/cmd/vaultgemma-server/main.go:29`

## Abstraction Benefits

1. **Separation of Concerns**: Each layer has a clear responsibility
2. **Testability**: Interfaces allow mocking at each boundary
3. **Flexibility**: Models can be swapped without changing Graph service code
4. **Scalability**: LocalAI can manage multiple models and backends independently
5. **Maintainability**: Changes to model loading don't affect workflow logic

## Component Relationships

```
┌─────────────────────────────────────────────────────────┐
│ Graph Service (LangGraph Workflows)                      │
│  - Orchestration chains                                  │
│  - Workflow state management                             │
│  - Knowledge graph integration                           │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP REST (OpenAI-compatible)
                   │
┌──────────────────▼──────────────────────────────────────┐
│ LocalAI Service (Domain Router & Model Manager)         │
│  - Domain detection & routing                            │
│  - Model provider abstraction                            │
│  - Backend selection                                     │
│  - Request processing                                    │
└──────────────────┬──────────────────────────────────────┘
                   │ Model Provider Interface
                   │
┌──────────────────▼──────────────────────────────────────┐
│ Model Layer (Physical Files & Loaders)                   │
│  - SafeTensors loader                                   │
│  - GGUF loader                                          │
│  - Transformers client                                  │
│  - Model registry                                       │
└──────────────────┬──────────────────────────────────────┘
                   │ File System / External Services
                   │
┌──────────────────▼──────────────────────────────────────┐
│ Physical Models                                         │
│  - models/vaultgemma-1b-transformers/                   │
│  - models/phi-3.5-mini-instruct-pytorch/                │
│  - models/gemma-7b-it-tensorrt/                         │
└──────────────────────────────────────────────────────────┘
```

## Related Documentation

- [ABSTRACTION_LAYERS.md](./ABSTRACTION_LAYERS.md) - Detailed explanation of each abstraction layer
- [DATA_FLOW.md](./DATA_FLOW.md) - Request/response flow with code references
- [MODEL_LOADING.md](./MODEL_LOADING.md) - Model discovery, loading, and caching
- [DIAGRAMS.md](./DIAGRAMS.md) - Visual diagrams of the architecture

