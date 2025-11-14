# VaultGemma LocalAI: Multi-Domain Agentic Inference Server

**Version**: 2.2.0  
**Status**: ✅ Production Ready  
**Last Updated**: November 14, 2025

## Recent Updates (v2.2.0)

- ✅ **Structured Logging**: High-performance zerolog integration with contextual logging
- ✅ **Optimized Caching**: LRU cache with TTL support and automatic cleanup
- ✅ **Connection Pooling**: Database and HTTP client pools for optimal resource usage

## Previous Updates (v2.1.0)

- ✅ **HANA Dependencies Removed**: Simplified architecture by removing SAP HANA integration
- ✅ **OpenTelemetry Tracing**: Built-in distributed tracing support with Jaeger exporter
- ✅ **API v2**: Enhanced API with workflow tracking, structured errors, and better metadata
- ✅ **Performance Optimizations**: Improved sorting algorithm using stdlib `sort.Slice()`
- ✅ **Codebase Cleanup**: Removed unused LocalAI directory (31MB savings)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Core Features](#2-core-features)
- [3. System Architecture](#3-system-architecture)
- [4. Getting Started](#4-getting-started)
- [5. Configuration](#5-configuration)
- [6. Testing](#6-testing)
- [7. Optional: Official llama.cpp Server](#7-optional-official-llamacpp-server)
- [8. HANA Integration](#8-hana-integration)
- [9. Contributing](#9-contributing)

---

## 1. Overview

VaultGemma LocalAI is a high-performance, production-ready inference server written entirely in Go. It is designed to host and manage a fleet of specialized AI agents (referred to as "domains") and serve them through an OpenAI-compatible API. 

Its core feature is an intelligent routing system that automatically directs incoming prompts to the most suitable agent based on the prompt's content. This allows for the creation of a powerful, multi-domain system where each agent is an expert in a specific area, such as SQL generation, blockchain transactions, or financial analysis.

This server is a key component of the AgenticAI ETH project, providing the core inference capabilities for the higher-level agentic systems.

## 2. Core Features

- **Multi-Domain Agent Routing**: Dynamically routes requests to over 24 specialized agent domains.
- **Intelligent Keyword Detection**: Automatically selects the best domain for a prompt by analyzing its content for keywords.
- **OpenAI-Compatible API**: Acts as a drop-in replacement for services like OpenAI, allowing easy integration with existing tools and applications.
- **Pure Go Implementation**: Built for performance and efficiency with no external dependencies like Python or C. Leverages native Go libraries for speed.
- **Production-Grade**: Includes essential production features such as rate limiting, CORS, health checks, Prometheus metrics, and request timeouts.
- **Vision/OCR Routing**: Integrates DeepSeek-OCR as a dedicated vision backend for image-to-text workflows.
- **Integrated Web UI**: Comes with a built-in web interface for easy testing, exploration, and real-time monitoring.
- **Distributed Tracing**: OpenTelemetry integration for request tracking across distributed systems.
- **API Versioning**: Both v1 (legacy) and v2 (enhanced) APIs available with backward compatibility.
- **Workflow Tracking**: Associate requests with workflows for better observability and debugging.
- **Structured Logging**: High-performance zerolog with contextual fields and specialized log methods.
- **Connection Pooling**: Optimized database and HTTP client pools with automatic management.
- **LRU Caching**: Intelligent cache with TTL support, automatic eviction, and performance tracking.

## 3. System Architecture

The server is designed with a modular architecture that separates concerns into distinct packages:

- **`cmd/vaultgemma-server`**: The main entry point of the application.
- **`pkg/server`**: Handles the HTTP server setup, API routing, and request/response lifecycle (e.g., health checks, metrics).
- **`pkg/domain`**: Manages the loading, configuration, and routing logic for the AI agent domains.
- **`config`**: Externalizes the agent domain definitions into a `domains.json` file, allowing for easy updates without code changes.
- **`web`**: Provides a static web UI for interacting with the server.
- **`internal`**: Contains supporting code and utilities not intended for external use.

When a request is received at the `/v1/chat/completions` endpoint, the server performs the following steps:

1.  **Request Validation**: Checks the request for correctness.
2.  **Domain Detection**: If the model is set to `"auto"`, the domain router analyzes the prompt's content and matches it against keywords registered for each domain.
3.  **Routing**: The request is forwarded to the designated agent domain.
4.  **Inference**: The agent's underlying model processes the prompt and generates a response.
5.  **Response**: The response is formatted to be OpenAI-compatible and sent back to the client.

## 4. Getting Started

### Prerequisites

- Go 1.21 or later
- Access to model files (to be placed in the `models` directory)

### Build and Run

1.  **Build the server binary:**
    ```bash
    go build -o bin/vaultgemma-server ./cmd/vaultgemma-server
    ```

2.  **Run the server:**
    ```bash
    ./bin/vaultgemma-server
    ```

The server will start on `http://localhost:8080` by default.

### Using the Web UI

Navigate to `http://localhost:8080` in your browser. The web UI provides:

- A real-time list of all loaded agent domains.
- A chat interface to send prompts and view responses.
- The ability to let the server auto-detect the domain or manually select one for testing.
- Live server statistics.

### Using the API

The server exposes several endpoints. The most important is the chat completions endpoint.

**Example: Auto-detect the domain**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Analyze environmental sustainability metrics for our portfolio"}],
    "max_tokens": 500
  }'
```

**Example: Manually specify a domain**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "0x5678-SQLAgent",
    "messages": [{"role": "user", "content": "SELECT * FROM users WHERE active = true"}],
    "max_tokens": 500
  }'
```

### Running the Transformers Sidecar

Domains configured with `"backend_type": "hf-transformers"` need the accompanying FastAPI sidecar to translate chat-style prompts into high-quality completions. A helper script bootstraps the environment and launches the service:

```bash
cd agenticAiETH_layer4_LocalAI
chmod +x scripts/start_transformers_sidecar.sh   # once
./scripts/start_transformers_sidecar.sh
```

The script creates a virtual environment (`.venv_transformers`), installs the dependencies listed in `requirements-transformers.txt`, and starts Uvicorn on `127.0.0.1:8081`. Adjust `TRANSFORMERS_CPU_HOST` or `TRANSFORMERS_CPU_PORT` if you need different bindings. You can verify the service is ready at `http://127.0.0.1:8081/health`.

With the sidecar running, LocalAI will automatically forward chat completion requests for transformer-backed domains (for example, `"model": "auto"` or `"model": "vaultgemma"`), returning natural language responses instead of placeholder token IDs.

> **Tip:** The sidecar depends on `torch`, `transformers`, and `accelerate`, so the first install can take a moment. Subsequent launches reuse the virtual environment.

### Optional: Code World Model (CWM) Fastgen backend

The repository vendors Meta's [Code World Model](https://github.com/facebookresearch/cwm) under `../../models/cwm`. To expose it as a LocalAI domain:

1. **Launch the Fastgen server** from the vendored checkout (requires GPUs that meet CWM's requirements):
   ```bash
   cd ../../models/cwm
   # Follow upstream instructions to download checkpoints.
   torchrun --nproc-per-node 2 -m serve.fgserve config=serve/configs/cwm.yaml \
     checkpoint_dir=/abs/path/to/cwm/checkpoint
   ```
2. **Enable the domain** when starting LocalAI:
   ```bash
   ENABLE_CWM_DOMAIN=1 ./bin/vaultgemma-server --config config/domains.json
   ```

When enabled, prompts routed to `0xC0DE-CodeWorldModelAgent` (or specifying `"model": "0xC0DE-CodeWorldModelAgent"`) will proxy through the Fastgen endpoint at `http://localhost:5678/v1/chat/completions`. The domain falls back to `phi-3.5-mini` if CWM is unavailable.

### Optional: CALM Continuous Reasoning domain

We also vendor **CALM** under `../../models/calm`. To expose the CALM-L checkpoint through the standard transformers sidecar:

1. Download the Hugging Face weights (accept the license first) and place them under `../../models/calm/hf/CALM-L` so the directory contains the usual `config.json`, `pytorch_model.bin`, tokenizer files, etc.
2. Ensure the transformers sidecar is running (`scripts/start_transformers_sidecar.sh`); it will now advertise the `calm-l` model.
3. Start LocalAI with `ENABLE_CALM_DOMAIN=1` so the domain loads from `config/domains.json`.

Requests routed to `0xCALM-ContinuousReasoningAgent` will call the transformers sidecar at `http://localhost:9090/v1/chat/completions`. If CALM is disabled or unavailable the server will automatically fall back to `phi-3.5-mini`.

## 5. Configuration

The server is configured through command-line flags and the `config/domains.json` file.

### Server Flags

- `--port`: The port for the server to listen on (default: `8080`).
- `--config`: The path to the domain configuration file (default: `config/domains.json`).

### Domain Configuration

To add, remove, or modify agent domains, edit the `config/domains.json` file. Each domain has a set of properties that define its behavior, including its name, the path to its model files, and the keywords that trigger it.

See the `config/README.md` for a detailed schema of the `domains.json` file.

## 6. Testing

### Unit and Integration Tests

To run the main test suite, which covers domain routing, API handlers, and other critical components:

```bash
go test -v ./tests/...
```

### Loader Benchmarks

The model loader exposes a micro-benchmark that exercises the chunked tensor reader. Run it with:

```bash
GOWORK=off go test -bench ReadFloat32Tensor ./pkg/models/ai
```

## 7. Optional: Official llama.cpp Server

For models already in `GGUF` format you can run the upstream C++ HTTP server that ships with `llama.cpp`. We vendored the repository under `third_party/llama.cpp` and provide a helper script to launch it.

### Build the server once

```bash
cd third_party/llama.cpp
cmake -B build -S . -DLLAMA_BUILD_SERVER=ON -DGGML_METAL=ON   # Metal backend on Apple Silicon
cmake --build build --target llama-server --config Release
```

This produces the binary `third_party/llama.cpp/bin/llama-server`.

### Launch with a GGUF model

```bash
cd agenticAiETH_layer4_LocalAI
scripts/run_llama_cpp_server.sh ../agenticAiETH_layer4_Models/phi/phi-3-pytorch-phi-3.5-mini-instruct-v2-q4_0.gguf \
  --port 8081 --ctx-size 4096
```

## 8. Distributed Tracing with OpenTelemetry

LocalAI includes built-in OpenTelemetry support for distributed tracing.

### Enable Tracing

Set the following environment variables:

```bash
export OTEL_TRACING_ENABLED=1
export OTEL_EXPORTER_JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

### Start Jaeger (Optional)

For local development, you can run Jaeger using Docker:

```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 14250:14250 \
  -p 9411:9411 \
  jaegertracing/all-in-one:latest
```

Then access the Jaeger UI at `http://localhost:16686`

### Trace Context Propagation

The v2 API supports W3C Trace Context propagation:

```bash
curl -X POST http://localhost:8080/v2/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}],
    "trace_parent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
  }'
```

## 9. API v2 - Enhanced Features

The v2 API provides enhanced features over the v1 API:

### Key Features

- **Structured Responses**: Consistent response format with comprehensive metadata
- **Workflow Tracking**: Associate requests with workflows using `workflow_id`
- **Enhanced Errors**: Detailed error codes with retry guidance
- **Built-in Tracing**: OpenTelemetry trace IDs in every response

### Example v2 Request

```bash
curl -X POST http://localhost:8080/v2/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Analyze SQL performance"}],
    "max_tokens": 512,
    "workflow_id": "sql-analysis-2024",
    "metadata": {
      "user_id": "user_123",
      "session_id": "session_456"
    }
  }'
```

### Example v2 Response

```json
{
  "id": "v2_req_1730000000000",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "0x5678-SQLAgent",
  "domain": "0x5678-SQLAgent",
  "status": "completed",
  "trace_id": "0af7651916cd43dd8448eb211c80319c",
  "workflow_id": "sql-analysis-2024",
  "choices": [...],
  "usage": {...},
  "metadata": {
    "latency_ms": 234,
    "backend_type": "gguf",
    "cache_hit": false
  }
}
```

For complete v2 API documentation, see [docs/API_V2.md](docs/API_V2.md)

## 9. Contributing

We welcome contributions to VaultGemma LocalAI. Please follow these guidelines:

1.  **Code Style:** Adhere to the standard Go formatting and linting practices.
2.  **Testing:** All new features and bug fixes must be accompanied by unit tests.
3.  **Pull Requests:** Create a pull request with a clear description of your changes.
