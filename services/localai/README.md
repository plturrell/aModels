# VaultGemma LocalAI: Multi-Domain Agentic Inference Server

**Version**: 2.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: October 28, 2025

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

### HANA Integration Tests

To run the HANA integration tests, you first need to set up your HANA environment:

1.  Copy `.env.hana.example` to `.env.hana` and update the credentials if necessary.
2.  Export the environment variables:

    ```bash
    export $(grep -v '^#' .env.hana | xargs)
    ```

3.  Run the tests with the `hana` build tag:

    ```bash
    go test -tags hana ./pkg/storage
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

## 8. HANA Integration

LocalAI can persist inference logs and caches to SAP HANA when built with the `hana` build tag.

1.  Ensure your HANA environment is configured as described in the "HANA Integration Tests" section.
2.  Start `vaultgemma-server` with the `hana` build tag to enable HANA-backed logging and caching:

    ```bash
    go build -tags hana -o bin/vaultgemma-server-hana ./cmd/vaultgemma-server
    ./bin/vaultgemma-server-hana
    ```

Without the tag, the package falls back to in-memory helpers so the service can run without a database connection.

## 9. Contributing

We welcome contributions to VaultGemma LocalAI. Please follow these guidelines:

1.  **Code Style:** Adhere to the standard Go formatting and linting practices.
2.  **Testing:** All new features and bug fixes must be accompanied by unit tests.
3.  **Pull Requests:** Create a pull request with a clear description of your changes.
