# Search Inference Service

This directory contains the search inference service, a specialized backend component responsible for executing AI models to power the platform's intelligent search features.

## 1. Overview

The search inference service is a core part of the AI-powered search experience. It is a dedicated service that hosts and runs machine learning models to perform complex tasks that go beyond traditional keyword matching. By offloading these computationally intensive tasks to a separate service, the main `server` can remain lightweight and focused on orchestration.

## 2. Core Responsibilities

- **Model Hosting**: Loads and serves deep learning models (e.g., Transformer-based models) in a production environment.
- **Semantic Search**: Takes a user's query and converts it into a vector embedding. It then uses this embedding to find semantically similar documents in a vector index.
- **Re-ranking**: Receives a list of candidate documents (e.g., from a keyword-based search) and uses a more powerful cross-encoder model to re-rank them based on their semantic relevance to the query.
- **Query Understanding**: Analyzes the user's query to understand its intent, identify key entities, and expand it with synonyms or related terms.
- **Performance Optimization**: This service is designed to run on specialized hardware, such as GPUs, to ensure that the AI models can be executed with low latency.

## 3. How It Works

The `server` communicates with the search inference service via an internal API. When an AI-powered feature is needed, the server sends a request to this service with the necessary data (e.g., the query and a list of documents). The inference service then runs the appropriate model and returns the result to the server.

## 4. How to Run

This service is a standalone application that needs to be run separately from the main server. It may have specific hardware and software requirements.

```bash
# Example command (actual command may vary)
./bin/search-inference --model-path ./models/semantic-ranker-v1 --port 9000 --gpu=true
```

## 5. Agent Catalog Endpoint

The search service exposes the live Agent SDK inventory via `GET /v1/agent-catalog`. The handler now returns an enriched payload that downstream dashboards can consume:

- `suites` / `tools` – raw suite/tool arrays, unchanged.
- `agent_catalog_summary` – human-readable overview of attached suites and tools.
- `agent_catalog_context` – multi-line context block suitable for prompt injection.
- `agent_catalog_stats` – counts for suites, implementations, unique tools, standalone tools, and the latest attach metadata.
- `agent_catalog_matrix` – runtime coverage grouped by implementation, including per-suite tool rollups.
- `agent_catalog_unique_tools` – flattened list of unique tool names.
- `agent_catalog_tool_details` – name/description pairs for standalone tools.

The helper `enrichSearchCatalog` (see `cmd/search-server/main.go`) converts the cached catalog into the shared enrichment format. UI clients can mirror the browser automation dashboard by rendering these fields directly.

Each refresh is also logged (look for `catalog:` entries in service logs) so telemetry aggregators can scrape the suite/tool counts without parsing HTTP responses.

For dashboards/alerting systems, `GET /v1/agent-catalog/stats` returns a compact JSON payload containing just the enrichment metadata and latest timestamp.

## 5. HANA Integration

The storage layer can persist documents, embeddings, and search telemetry in SAP HANA.
The HANA-backed implementations are protected by the `hana` build tag so the service
continues to build without a database connection by default.

1. Copy `.env.hana.example` to `.env.hana` and adjust credentials if required.
2. Export the variables before running the service or tests:

   ```bash
   export $(grep -v '^#' .env.hana | xargs)
   ```

3. Run storage checks with the HANA tag:

   ```bash
   go test -tags hana ./pkg/storage
   ```

4. Start the search inference server with the same environment in place.

Without the `hana` tag the package falls back to lightweight stubs so the rest of the
search pipeline can be exercised without a database connection.
