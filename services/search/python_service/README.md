# FastAPI Search Gateway

This FastAPI application orchestrates the Go-based embedding service while
persisting search data into a canonical Elasticsearch backend, with optional
Redis caching and SAP HANA health checks.

## Features

- Always persists documents into Elasticsearch (when reachable) and stores the
  generated embeddings alongside metadata.
- Leverages the Go search server for embedding generation when `GO_SEARCH_URL`
  is provided, but no longer depends on it for persistence.
- Provides an in-memory fallback when Elasticsearch is unavailable so the API
  can still accept writes, albeit without durability.
- Optional Redis caching for search responses.
- Health endpoint exposing the live status of each connector.

## Configuration

Set the following environment variables as needed:

- `GO_SEARCH_URL` – URL of the Go search inference server (e.g. `http://localhost:7070`).
- `ELASTICSEARCH_URLS` – Comma-separated list of Elasticsearch hosts (required
  for canonical persistence).
- `ELASTICSEARCH_API_KEY` or `ELASTICSEARCH_USERNAME`/`ELASTICSEARCH_PASSWORD`.
- `ELASTICSEARCH_INDEX` – Target index name (defaults to `agenticaieth-docs`).
- `REDIS_URL` – Redis connection string (optional).
- `HANA_DSN` – SAP HANA DSN (optional, used for health reporting only).
- `PORT` – HTTP port for the FastAPI server (defaults to `8080`).

## Usage

```bash
cd agenticAiETH_layer4_Search/python_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

During development you can point the service at the local Go search server by
setting `GO_SEARCH_URL=http://localhost:7070`; this is optional but recommended
to obtain high-quality embeddings for vector search.
