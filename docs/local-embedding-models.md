# Local Embedding Models Configuration

This document lists all locally available embedding models and ensures Elasticsearch and langextract-api use only local models.

## Available Local Embedding Models

### 1. **all-MiniLM-L6-v2** (sentence-transformers)
- **Status**: ✅ Currently configured and in use
- **Location**: Downloaded automatically by transformers service
- **Configuration**: 
  - `TRANSFORMERS_MODEL=all-MiniLM-L6-v2` in docker-compose
  - Used by: LocalAI service, search service, training scripts
- **Model Card**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Usage**: Vector embeddings for semantic search, similarity matching

### 2. **Training Configuration Models**
- **sentence-transformers/all-MiniLM-L6-v2**: Used in relational transformer training
- **Location**: `config/training/rt_sgmi.yaml` (line 55)
- **Usage**: Text encoding for relational transformer training

## Service Configuration

### LocalAI Service (Embeddings)
- **Service**: `transformers-service` (port 9090)
- **Model**: `all-MiniLM-L6-v2`
- **Configuration**: `TRANSFORMERS_MODEL=all-MiniLM-L6-v2`
- **Endpoint**: `http://transformers-service:9090`
- **Status**: ✅ Uses local models only

### Search Service (Embeddings)
- **Service**: `search-python` (port 8091)
- **Model**: Uses transformers service via `TRANSFORMERS_BASE_URL`
- **Configuration**: 
  - `TRANSFORMERS_BASE_URL=http://search-python:8091`
  - `TRANSFORMERS_MODEL=all-MiniLM-L6-v2`
- **Status**: ✅ Uses local models only (via transformers service)

### Elasticsearch Plugin (Embeddings)
⚠️ **REQUIRES CONFIGURATION**: Elasticsearch plugin code has references to external APIs but must be configured to use local models only.

**Current State**:
- Elasticsearch plugin has inference service code for OpenAI, Anthropic, DeepSeek APIs
- These are third-party plugin code (`infrastructure/third_party/elasticsearch`)
- **Action Required**: Ensure Elasticsearch is configured to use only local inference services

**Configuration Needed**:
1. Disable external API inference services in Elasticsearch
2. Configure Elasticsearch to use local inference endpoint (e.g., `search-python:8091`)
3. Or use Elasticsearch's built-in embedding models (if available locally)

**Recommended Configuration**:
```yaml
# Elasticsearch inference service should point to local service
inference:
  services:
    local_embeddings:
      service: http://search-python:8091
      model: all-MiniLM-L6-v2
```

### langextract-api (Embeddings)
⚠️ **REQUIRES CONFIGURATION**: langextract-api supports external models but must be configured to use LocalAI only.

**Current State**:
- langextract-api supports OpenAI, Anthropic, Gemini, and Ollama
- Can be configured to use local models via Ollama or LocalAI

**Configuration Needed**:
1. Configure langextract-api to use LocalAI endpoint
2. Or disable langextract-api entirely (set `LANGEXTRACT_API_URL=` empty)

**Recommended Configuration**:
```bash
# Option 1: Use LocalAI (if langextract supports OpenAI-compatible API)
LANGEXTRACT_API_URL=http://localai:8081/v1
LANGEXTRACT_API_KEY=not-needed

# Option 2: Disable langextract-api
LANGEXTRACT_API_URL=
LANGEXTRACT_API_KEY=
```

## Verification Checklist

- [x] LocalAI transformers service uses `all-MiniLM-L6-v2` (local)
- [x] Search service uses transformers service (local)
- [ ] Elasticsearch plugin configured to use local inference only
- [ ] langextract-api configured to use LocalAI or disabled

## Summary

**Local Embedding Models Available:**
- ✅ `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

**Services Using Local Embeddings:**
- ✅ LocalAI service (via transformers-service)
- ✅ Search service (via transformers-service)
- ⚠️ Elasticsearch plugin (needs configuration)
- ⚠️ langextract-api (needs configuration)

**Action Items:**
1. Configure Elasticsearch to use local inference endpoint only
2. Configure langextract-api to use LocalAI or disable it
3. Document any additional embedding models if added to the codebase

