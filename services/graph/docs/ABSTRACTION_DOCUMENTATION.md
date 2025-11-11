# Graph-LocalAI-Models Abstraction Documentation

This directory contains comprehensive documentation explaining the abstraction layers between the Graph service, LocalAI service, and underlying models.

## Documentation Overview

### [DIAGRAMS.md](./DIAGRAMS.md)
Visual diagrams in Mermaid format showing:
- System architecture
- Abstraction layers
- Request flow sequence
- Model loading flow
- Component interactions

### [ARCHITECTURE.md](./ARCHITECTURE.md)
High-level system overview covering:
- Three-layer architecture (Graph → LocalAI → Models)
- Component relationships
- Key interfaces
- Configuration points
- Abstraction benefits

### [ABSTRACTION_LAYERS.md](./ABSTRACTION_LAYERS.md)
Detailed breakdown of each abstraction layer:
- Graph Service → LocalAI interface
- LocalAI internal abstractions (Domain, Model Provider, Backend Provider)
- Model loading abstraction
- Code references for each layer

### [DATA_FLOW.md](./DATA_FLOW.md)
Complete request/response flow tracing:
- Step-by-step flow from Graph workflow to model inference
- Code references at each stage
- Error handling flow
- Performance considerations

### [MODEL_LOADING.md](./MODEL_LOADING.md)
Model lifecycle documentation:
- Model discovery from configuration
- Model registration and lazy loading
- Caching mechanisms (in-memory and persistent)
- Backend selection logic
- GPU allocation

## Quick Start

1. **Start with [ARCHITECTURE.md](./ARCHITECTURE.md)** for a high-level understanding
2. **Review [DIAGRAMS.md](./DIAGRAMS.md)** for visual representation
3. **Read [ABSTRACTION_LAYERS.md](./ABSTRACTION_LAYERS.md)** for detailed interface documentation
4. **Follow [DATA_FLOW.md](./DATA_FLOW.md)** to understand request processing
5. **Explore [MODEL_LOADING.md](./MODEL_LOADING.md)** for model management details

## Key Concepts

### Abstraction Layers

1. **Graph Service Layer**: Workflow orchestration using LangGraph
   - Abstracts away model details
   - Uses LocalAI as HTTP service
   - Standard OpenAI-compatible interface

2. **LocalAI Service Layer**: Domain routing and model management
   - Domain-based routing (24+ specialized agents)
   - Model provider abstraction
   - Backend selection (SafeTensors, GGUF, Transformers)

3. **Models Layer**: Physical model files and loaders
   - Multiple formats (SafeTensors, GGUF, Transformers)
   - Lazy loading for performance
   - Caching for efficiency

### Key Interfaces

- **Graph → LocalAI**: HTTP REST (OpenAI-compatible)
- **ModelProvider**: Unified interface for all model types
- **BackendProvider**: Backend selection abstraction
- **ModelLoader**: Format-agnostic model loading

## Code Locations

### Graph Service
- `services/graph/pkg/workflows/orchestration_processor.go` - Orchestration chain creation
- `infrastructure/third_party/orchestration/llms/localai/localai.go` - LocalAI client

### LocalAI Service
- `services/localai/pkg/server/vaultgemma_server.go` - HTTP server
- `services/localai/pkg/domain/domain_config.go` - Domain management
- `services/localai/pkg/server/interfaces.go` - Abstraction interfaces
- `services/localai/pkg/models/model_loader.go` - Model loading
- `services/localai/pkg/server/model_cache.go` - Model caching

### Models
- `models/` - Physical model files directory
- `services/localai/config/domains.json` - Domain configuration

## Related Documentation

- [Graph Service README](../README.md) - Graph service overview
- [LocalAI README](../../localai/README.md) - LocalAI service overview
- [Integration Documentation](../INTEGRATION.md) - Service integration details

