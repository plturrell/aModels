# LocalAI Model Integration & Optimization Report

## Overview
This report documents which models are enabled, integrated, and optimized with LocalAI based on the current configuration and codebase analysis.

---

## Model Backend Types

LocalAI supports three main backend types:
1. **SafeTensors** (`safetensors`) - Pure Go implementation, CPU-based
2. **GGUF** (`gguf`) - Quantized models via go-llama.cpp, CPU/GPU
3. **HuggingFace Transformers** (`hf-transformers`) - Python service, CPU/GPU

---

## Models in `/home/aModels/models` Directory

### Available Model Directories:
1. **vaultgemma-1b-transformers/** - SafeTensors format
2. **phi-3.5-mini-instruct-pytorch/** - HuggingFace Transformers format
3. **granite-4.0-h-micro-transformers/** - HuggingFace Transformers format
4. **gemma-2b-it-tensorrt/** - TensorRT optimized (contains GGUF)
5. **gemma-7b-it-tensorrt/** - TensorRT optimized (contains GGUF)
6. **DeepSeek-OCR/** - OCR/Vision model
7. **sentencepiece/** - Tokenizer library
8. **glove/** - Embeddings model
9. **open_deep_research/** - Research model
10. **TinyRecursiveModels/** - Specialized models
11. **sap-rpt-1-oss-main/** - SAP model

---

## Enabled Models (from `domains.json`)

### Production Configuration (`domains.production.json`)

| Domain ID | Model Name | Backend | Model Path | Status |
|-----------|------------|---------|------------|--------|
| `gemma-2b-it` | Gemma 2B-it | GGUF | `../../models/gemma-2b-it-tensorrt/2b-it/model.gguf` | ✅ Enabled |
| `phi-3.5-mini` | Phi-3.5-mini | hf-transformers | `../../models/phi-3.5-mini-instruct-pytorch` | ✅ Enabled |
| `granite-4.0` | IBM Granite 4.0 | hf-transformers | `../../models/granite-4.0-h-micro-transformers` | ✅ Enabled |
| `gemma-7b-it` | Gemma 7B-it | GGUF | `../../models/gemma-7b-it-tensorrt/7b-it/model.gguf` | ✅ Enabled |
| `vaultgemma` | VaultGemma | safetensors | `../../models/vaultgemma-1b-transformers` | ✅ Enabled |
| `general` | General Assistant | safetensors | `../../models/vaultgemma-1b-transformers` | ✅ Enabled (default) |

### Development Configuration (`domains.json`)

**Total: 30+ domain configurations**

#### Key Models by Backend:

**GGUF Models (Quantized):**
- `gemma-2b-q4_k_m.gguf` - Used by 20+ domains
- `gemma-7b-q4_k_m.gguf` - Used by BrowserAnalysisAgent

**HuggingFace Transformers:**
- `phi-3.5-mini` - Used by VectorProcessingAgent, general domain, and as fallback
- `granite-4.0` - Used by SubledgerAgent, ESGFinanceAgent
- `gemma-2b-it` - Used by GemmaAssistant (0xG2B)
- `gemma-7b-it` - Used by Gemma7BAssistant (0xG7B) - **Conditional** (requires `ENABLE_GEMMA7B` env var)

**SafeTensors:**
- `vaultgemma-1b-transformers` - Default fallback model

---

## Integration Details

### 1. SafeTensors Backend (Pure Go)

**Models:**
- VaultGemma 1B

**Integration:**
- Loaded via `ai.LoadVaultGemmaFromSafetensors()`
- Supports lazy loading via `ModelCache`
- Default fallback for all domains
- CPU-only inference

**Optimization:**
- Pure Go implementation (no external dependencies)
- Memory-mapped loading
- Configurable via `MODEL_CACHE_MAX_MEMORY_MB` (default: 8GB)

**Code Location:**
- `/home/aModels/services/localai/pkg/models/ai/`
- `/home/aModels/services/localai/pkg/server/model_cache.go`

---

### 2. GGUF Backend (Quantized Models)

**Models:**
- Gemma 2B-it (TensorRT optimized)
- Gemma 7B-it (TensorRT optimized)
- All domains using `.gguf` files

**Integration:**
- Loaded via `gguf.Load()` using go-llama.cpp
- Supports lazy loading
- GPU acceleration available

**Optimization:**
- **GPU Acceleration:**
  - Auto-detected if `CUDA_VISIBLE_DEVICES` or `GGML_CUDA` is set
  - Can be disabled with `DISABLE_GGUF_GPU=1`
  - Offloads all layers to GPU (`gpuLayers = -1`)
- **Quantization:** Q4_K_M (4-bit quantization)
- **Memory Mapping:** Enabled by default (`SetMMap(true)`)
- **Context Size:** 2048 tokens
- **Batch Size:** 512

**Code Location:**
- `/home/aModels/services/localai/pkg/models/gguf/model.go`
- `/home/aModels/services/localai/pkg/server/model_cache.go` (lines 232-260)

**Configuration:**
```go
// GPU layers detection
if os.Getenv("CUDA_VISIBLE_DEVICES") != "" || os.Getenv("GGML_CUDA") != "" {
    gpuLayers = -1  // Offload all layers to GPU
}
```

---

### 3. HuggingFace Transformers Backend

**Models:**
- Phi-3.5-mini-instruct
- Granite-4.0-h-micro
- Gemma-2b-it
- Gemma-7b-it

**Integration:**
- Python service at `http://transformers-service:9090/v1/chat/completions`
- Client created via `transformers.NewClient()`
- HTTP-based communication

**Optimization:**
- **GPU Support:**
  - Uses `torch.float16` on GPU (CUDA)
  - Uses `torch.float32` on CPU
  - Device mapping: `cuda:0` or `cpu`
- **Memory Optimization:**
  - `low_cpu_mem_usage=True`
  - Model caching with `@lru_cache`
- **Quantization Options:**
  - BitsAndBytesConfig (4-bit, 8-bit)
  - XPU quantization (4-bit, 8-bit)
  - Configurable via request parameters

**Code Location:**
- `/home/aModels/services/localai/services/transformers_cpu_server.py`
- `/home/aModels/services/localai/pkg/transformers/client.go`

**Service Configuration:**
```python
# GPU optimization
if DEVICE == "cuda":
    torch_dtype = torch.float16
    device_map = "cuda:0"
else:
    torch_dtype = torch.float32
    device_map = "cpu"
```

---

## Model Resolution & Fallback Logic

### Resolution Priority:
1. Domain-specific model (from domain config)
2. Fallback model (if specified in config)
3. Default domain model (`general` or `vaultgemma`)

### Backend Selection:
1. `TRANSFORMERS_BASE_URL` → Transformers backend
2. `GGUF_ENABLE=1` → GGUF backend
3. Default → SafeTensors backend

**Code:** `/home/aModels/services/localai/pkg/server/chat_helpers.go` (lines 127-215)

---

## Optimization Features

### 1. Lazy Loading
- Models loaded on first use
- Controlled by `ENABLE_LAZY_LOADING` env var
- Reduces startup time
- Memory-efficient

### 2. Model Caching
- `ModelCache` manages loaded models
- Memory limit: 8GB (configurable via `MODEL_CACHE_MAX_MEMORY_MB`)
- LRU eviction based on access time
- Prevents duplicate loading

### 3. GPU Acceleration

**GGUF:**
- Auto-detected GPU support
- All layers offloaded to GPU
- Configurable via environment variables

**Transformers:**
- Float16 precision on GPU
- Automatic device mapping
- CUDA support

### 4. Quantization

**GGUF Models:**
- Q4_K_M quantization (4-bit)
- Reduces memory footprint
- Maintains quality

**Transformers:**
- BitsAndBytesConfig support
- 4-bit and 8-bit quantization
- XPU quantization for Intel GPUs

### 5. Memory Management
- Memory-mapped files (MMap) for GGUF
- Low CPU memory usage for Transformers
- Configurable memory limits
- Automatic cleanup

---

## Domain-Specific Configurations

### High-Performance Domains:
- **BrowserAnalysisAgent** (`0xBR0W`): Uses Gemma 7B (GGUF)
- **SubledgerAgent** (`0x5D1A`): Uses Granite 4.0 (Transformers)
- **ESGFinanceAgent** (`0xF3C0`): Uses Granite 4.0 (Transformers)

### Lightweight Domains:
- Most Layer 1-3 agents use Gemma 2B (GGUF, quantized)
- VectorProcessingAgent uses Phi-3.5-mini (Transformers)

### Conditional Models:
- **Gemma 7B Assistant** (`0xG7B`): Requires `ENABLE_GEMMA7B=1`

---

## Performance Optimizations Summary

| Backend | GPU Support | Quantization | Memory Optimization | Lazy Loading |
|---------|-------------|--------------|---------------------|--------------|
| SafeTensors | ❌ | ❌ | ✅ MMap | ✅ |
| GGUF | ✅ Auto | ✅ Q4_K_M | ✅ MMap | ✅ |
| Transformers | ✅ CUDA | ✅ 4/8-bit | ✅ Low CPU Mem | ❌ (Service-based) |

---

## Environment Variables for Optimization

```bash
# Lazy Loading
ENABLE_LAZY_LOADING=1

# Model Cache
MODEL_CACHE_MAX_MEMORY_MB=8192

# GGUF GPU
CUDA_VISIBLE_DEVICES=0
GGML_CUDA=1
DISABLE_GGUF_GPU=0  # Set to 1 to disable

# Transformers
TRANSFORMERS_BASE_URL=http://transformers-service:9090
DEVICE=cuda  # or cpu

# Conditional Models
ENABLE_GEMMA7B=1

# Backend Selection
GGUF_ENABLE=1
```

---

## Model Status by Domain

### ✅ Fully Integrated & Optimized:
1. **VaultGemma** - SafeTensors, lazy loading, memory-mapped
2. **Gemma 2B-it** - GGUF, GPU-accelerated, quantized
3. **Gemma 7B-it** - GGUF, GPU-accelerated, quantized
4. **Phi-3.5-mini** - Transformers, GPU-optimized, float16
5. **Granite-4.0** - Transformers, GPU-optimized, float16

### ⚠️ Partially Integrated:
- **DeepSeek-OCR** - Vision/OCR service (separate integration)
- **TinyRecursiveModels** - Not in domain configs
- **sap-rpt-1-oss** - Not in domain configs

### ❌ Not Integrated:
- **glove/** - Embeddings (not LLM)
- **sentencepiece/** - Tokenizer library (supporting)
- **open_deep_research/** - Empty/unused

---

## Recommendations

1. **Enable GPU for GGUF models** - Set `CUDA_VISIBLE_DEVICES` or `GGML_CUDA`
2. **Use lazy loading** - Set `ENABLE_LAZY_LOADING=1` for faster startup
3. **Configure memory limits** - Adjust `MODEL_CACHE_MAX_MEMORY_MB` based on available RAM
4. **Enable Gemma 7B** - Set `ENABLE_GEMMA7B=1` if needed for high-performance domains
5. **Monitor GPU usage** - Check GPU utilization for Transformers and GGUF backends

---

## Summary

**Total Models Enabled:** 6 core models (production config)
**Total Domains Configured:** 30+ domains (development config)
**Backend Types:** 3 (SafeTensors, GGUF, Transformers)
**Optimization Features:** Lazy loading, GPU acceleration, quantization, memory management

All major models are properly integrated with optimization features enabled. The system supports both CPU and GPU inference with automatic backend selection and fallback mechanisms.

