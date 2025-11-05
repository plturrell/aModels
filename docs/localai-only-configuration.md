# LocalAI-Only Configuration

This document confirms that all services in the aModels stack are configured to use **only LocalAI** for LLM operations, with no external LLM dependencies (OpenAI, Anthropic, Gemini, etc.).

## Service Configuration Status

### ✅ DeepAgents Service
- **Status**: Fully configured for LocalAI-only operation
- **Configuration**: 
  - `LOCALAI_URL=http://localai:8081` (or `http://localai:8080` in base compose)
  - `LOCALAI_MODEL=general` (defaults to "general" domain which uses phi-3.5-mini)
  - `ANTHROPIC_API_KEY=` (empty, disabled)
  - `OPENAI_API_KEY=` (empty, disabled)
- **Implementation**: Uses LangChain `ChatOpenAI` with `base_url` pointing to LocalAI
- **Location**: `services/deepagents/agent_factory.py`

### ✅ Graph Service
- **Status**: Configured for LocalAI-only operation
- **Configuration**:
  - `LOCALAI_URL=http://localai:8081` (added to docker-compose.yml)
  - Defaults to `http://localai:8080` if not set
- **Implementation**: Uses `LOCALAI_URL` environment variable for LLM operations
- **Location**: `services/graph/pkg/workflows/unified_processor.go`, `services/graph/pkg/workflows/orchestration_processor.go`

### ✅ Gateway Service
- **Status**: Configured for LocalAI-only operation
- **Configuration**:
  - `LOCALAI_URL=http://localai:8080`
- **Implementation**: Uses LocalAI for health checks and chat completions
- **Location**: `services/gateway/main.py`

### ⚠️ Extract Service
- **Status**: Partially configured - uses langextract-api (can be disabled)
- **Configuration**:
  - `LANGEXTRACT_API_URL` (can be empty to disable)
  - `LANGEXTRACT_API_KEY` (can be empty)
  - Note: langextract-api is an optional external service for extraction operations
  - If not needed, set `LANGEXTRACT_API_URL=` (empty) to disable
- **Default Model Reference**: References `gemini-2.5-flash` but only used if langextract-api is enabled
- **Location**: `services/extract/main.go`, `services/extract/internal/config/config.go`

### ⚠️ AgentFlow Service (LangFlow Integration)
- **Status**: Uses LangFlow (can be external or local)
- **Configuration**:
  - `AGENTFLOW_LANGFLOW_URL` (defaults to `http://localhost:7860`)
  - `AGENTFLOW_LANGFLOW_API_KEY` (optional)
  - `AGENTFLOW_LANGFLOW_AUTH_TOKEN` (optional)
- **Note**: LangFlow is a workflow orchestration service that can run locally or externally. It can be configured to use LocalAI as its backend.
- **Location**: `services/agentflow/service/main.py`, `services/agentflow/service/config.py`

### ✅ Search Service
- **Status**: No direct LLM dependencies
- **Note**: Search service uses Elasticsearch/OpenSearch for indexing and retrieval, not LLM operations

### ✅ Training Service
- **Status**: No direct LLM dependencies
- **Note**: Training scripts use local models and don't require external LLM services

## Docker Compose Configuration

### Base Compose (`infrastructure/docker/compose.yml`)
- DeepAgents: `ANTHROPIC_API_KEY=` and `OPENAI_API_KEY=` are empty (disabled)
- Gateway: `LOCALAI_URL=http://localai:8080`

### Brev Compose (`infrastructure/docker/brev/docker-compose.yml`)
- DeepAgents: `ANTHROPIC_API_KEY=` and `OPENAI_API_KEY=` are empty (disabled)
- Graph: `LOCALAI_URL=http://localai:8081` (added)
- Extract: `LANGEXTRACT_API_URL` and `LANGEXTRACT_API_KEY` can be empty (disabled)

## LocalAI Service Configuration

The LocalAI service is configured with:
- **Port**: 8080 (internal), 8081 (external)
- **Config**: `services/localai/config/domains.json`
- **Transformers Backend**: `http://transformers-service:9090` for hf-transformers models
- **GGUF Backend**: Direct model loading from `/models/` directory

All agent domains are configured in `domains.json` with:
- Docker service names for transformers backend (`transformers-service:9090`)
- Absolute paths for GGUF models (`/models/...`)
- No external API dependencies

## Verification Checklist

- [x] DeepAgents uses only LocalAI (no external API keys)
- [x] Graph service has LOCALAI_URL configured
- [x] Gateway service uses LocalAI
- [x] Extract service langextract-api can be disabled
- [x] Search service has no LLM dependencies
- [x] Training service has no LLM dependencies
- [ ] AgentFlow/LangFlow can be configured to use LocalAI (optional)
- [x] All services use Docker service names (not localhost)

## Notes

1. **LangFlow**: If you want AgentFlow to use LocalAI, you need to configure LangFlow itself to use LocalAI as its backend. This is a LangFlow configuration, not an aModels configuration.

2. **langextract-api**: This is an optional service for document extraction. If you want pure LocalAI-only operation, you can disable it by setting `LANGEXTRACT_API_URL=` (empty).

3. **Port Differences**: The base compose uses `localai:8080` (internal port) while the Brev compose uses `localai:8081` (external port). Both work correctly within Docker networks.

## Allowed Models

The codebase is configured to use **only** the following models:

1. **LocalAI Models** (via LocalAI service)
   - All agent domains configured in `domains.json`
   - GGUF models (Gemma, Phi, etc.)
   - HF-Transformers models (phi-3.5-mini, granite-4.0, gemma-2b-it, gemma-7b-it)

2. **Embedding Models** (in codebase - local only)
   - `all-MiniLM-L6-v2` (sentence-transformers/all-MiniLM-L6-v2)
   - **Dimensions**: 384
   - **Location**: Downloaded automatically by transformers service
   - **Configuration**: `TRANSFORMERS_MODEL=all-MiniLM-L6-v2` in docker-compose
   - **Used by**: 
     - LocalAI service (via transformers-service:9090)
     - Search service (via search-python:8091)
     - Training scripts (relational transformer)
   - ⚠️ **Elasticsearch plugin**: Must be configured to use local inference only
   - ⚠️ **langextract-api**: Must be configured to use LocalAI or disabled

3. **DeepSeek OCR** (in codebase)
   - Vision model for OCR operations
   - Backend type: `deepseek-ocr`
   - Configured via: `DEEPSEEK_OCR_SCRIPT` environment variable
   - Default: `./scripts/deepseek_ocr_cli.py`
   - Model: `deepseek-ai/DeepSeek-OCR` (HuggingFace)
   - Used by extract service for document OCR

## Summary

All core services (DeepAgents, Graph, Gateway) are configured to use **only LocalAI** with no external LLM dependencies. The Extract service can optionally use langextract-api, but this can be disabled. AgentFlow uses LangFlow which can be configured separately to use LocalAI.

**Allowed Models Only:**
- ✅ LocalAI (all agent domains)
- ✅ Embedding models (all-MiniLM-L6-v2)
- ✅ DeepSeek OCR (for vision/OCR tasks)
- ❌ No OpenAI, Anthropic, Gemini, or other external LLM services

