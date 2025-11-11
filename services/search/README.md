# Agentic Search & Discovery Platform

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Core Features](#2-core-features)
- [3. Architecture](#3-architecture)
- [4. Getting Started](#4-getting-started)
- [5. Directory Reference](#5-directory-reference)
- [6. Dependencies](#6-dependencies)
- [7. Testing](#7-testing)
- [8. Contributing](#8-contributing)

---

## 1. Overview

This project is a next-generation search and discovery platform designed to provide intelligent, AI-powered search capabilities. It goes beyond traditional keyword matching by incorporating semantic understanding, question-answering, and other advanced machine learning techniques to deliver more relevant and contextual results.

The platform is architected as a comprehensive, standalone system with distinct client and server components, a modular architecture, and dedicated modules for handling complex AI operations like inference and embedding generation.

## 2. Core Features

- **Hybrid Search**: Combines traditional keyword-based (lexical) search with modern vector-based (semantic) search to get the best of both worlds.
- **Question Answering (QA)**: A dedicated `qa` module allows users to ask questions in natural language and receive direct answers, rather than just a list of links.
- **Semantic Inference**: The `search-inference` engine uses deep learning models to understand the intent and context behind a user's query.
- **Embeddings Support**: A full-featured `embeddings` component for generating and managing the vector representations of data that power semantic search.
- **Modular Architecture**: The `modules` directory allows for the extension of the platform's core functionality with new features and integrations.
- **Client-Server Model**: A clear separation between the `server` (backend) and `client` (frontend) makes the platform scalable and flexible.
- **Extended Features (`x-pack`)**: A dedicated pack for advanced or commercial features, such as security, monitoring, and reporting.

## 3. Architecture

The platform is structured as a multi-component system:

- **`server`**: The core backend application that handles API requests, orchestrates search queries, and communicates with the other components.
- **`client`**: The frontend user interface for interacting with the search platform.
- **`search-inference`**: A specialized service responsible for executing AI models to perform tasks like semantic ranking and query understanding.
- **`embeddings`**: A service or library for creating vector embeddings from text data.
- **`modules`**: A collection of pluggable modules that extend the core functionality.
- **`qa`**: The question-answering system that provides direct answers to user queries.
- **`python_service`**: A Python-based service that provides additional AI capabilities and integration with Python-based libraries.
- **`libs`**: Shared libraries and code used by multiple components of the platform.

## 4. Getting Started

To get the search platform up and running, follow these steps:

1.  **Install Dependencies:** Make sure you have the necessary dependencies installed (see the Dependencies section below).
2.  **Build the Components (optional when using Docker Compose):**

    ```bash
    go build ./server
    npm install --prefix client && npm run build --prefix client
    ```

3.  **Run the Services:**

    **Option A – Docker Compose (recommended for local development):**

    ```bash
    docker compose up --build
    ```

    This brings up Elasticsearch, Redis, LocalAI, the Go search-inference service (port `8090`), and the FastAPI gateway (port `8081`).

    **Option B – Manual startup:**

    ```bash
    ./server/server &
    python ./python_service/app.py &
    ```

4.  **Access the Services:**

    - Search inference API: `http://localhost:8090`
    - FastAPI gateway: `http://localhost:8081`
    - Elasticsearch: `http://localhost:9200`
    - LocalAI: `http://localhost:8080` (internal to Docker network)

5.  **OpenAPI Documentation:**

    The FastAPI gateway exports an OpenAPI document at `python_service/openapi.json`. Regenerate it after API changes by running:

    ```bash
    cd python_service
    python - <<'PY'
    import json
    from app import app

    with open("openapi.json", "w") as f:
        json.dump(app.openapi(), f, indent=2)
    PY
    ```

## 5. Directory Reference

For detailed information on each component, please see the `README.md` file located in the corresponding subdirectory.

## 6. Dependencies

- **Go:** Version 1.20 or higher
  - The search-inference service now requires Go **1.21** (see `search-inference/go.mod`).
- **Python:** Version 3.10 or higher
- **Node.js:** Version 18 or higher

## 7. Testing

### Unit Tests

**Python Service:**
```bash
cd python_service
pytest test_app.py -v
```

**Go Service:**
```bash
cd search-inference
go test ./...
```

### Integration Tests

Integration tests require the services to be running via Docker Compose:

```bash
# Start services
docker compose up -d

# Wait for services to be ready (about 30 seconds)
sleep 30

# Run integration tests
cd python_service
export AUTH_ENABLED=false  # Or set to true and provide TEST_API_KEY
export TEST_API_KEY=your-api-key  # If AUTH_ENABLED=true
pytest test_integration.py -v

# Stop services
docker compose down
```

### Test Configuration

Integration tests can be configured via environment variables:

- `PYTHON_SERVICE_URL`: Python service URL (default: http://localhost:8081)
- `GO_SERVICE_URL`: Go service URL (default: http://localhost:8090)
- `AUTH_ENABLED`: Enable authentication in tests (default: false)
- `TEST_API_KEY`: API key for authenticated tests

## 8. Contributing

We welcome contributions to the Agentic Search & Discovery Platform. Please follow these guidelines:

1.  **Code Style:** Adhere to the standard formatting and linting practices for each language.
2.  **Testing:** All new features and bug fixes must be accompanied by unit tests.
3.  **Pull Requests:** Create a pull request with a clear description of your changes.
