# FastAPI Gateway for Postgres Lang Service

This module exposes a REST/JSON interface over the existing `PostgresLangService` gRPC API so browser-based tools and other HTTP clients can reuse the telemetry data stored in Postgres.

## Features

- `/health` – service and database health metadata
- `/operations` – list and filter recorded operations (pagination, time windows, library/session filters)
- `/operations/{id}` – fetch a single operation
- `/operations` (POST) – log a new operation document
- `/analytics` – aggregated operation statistics
- `/cleanup` – remove older records
- `/statuses` – discover allowed operation status names
- `/db/*` – optional Postgres admin endpoints (table listing & SQL execution)

## Prerequisites

- Python 3.9+
- Access to the running gRPC service (`POSTGRES_LANG_SERVICE_ADDR`, defaults to `localhost:50055`)

Install dependencies (the repo uses `requirements.txt` for convenience):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
export POSTGRES_LANG_SERVICE_ADDR="localhost:50055"
export POSTGRES_LANG_GATEWAY_CORS="http://localhost:9222"
python -m gateway
```

To expose the Postgres admin interface, supply a connection string (falls back to `POSTGRES_DSN` when set):

```bash
export POSTGRES_LANG_DB_DSN="postgres://user:pass@localhost:5432/lang_ops?sslmode=disable"
# optional overrides
export POSTGRES_DB_ALLOW_MUTATIONS=false   # leave false to stay read-only
export POSTGRES_DB_DEFAULT_LIMIT=200       # max rows returned when no LIMIT is supplied
```

The service listens on `0.0.0.0:8000` by default – override with `FASTAPI_HOST` / `FASTAPI_PORT`. Set `FASTAPI_RELOAD=true` for development auto reloads.

## Notes

- All timestamps accept ISO-8601 strings (e.g. `2025-01-24T12:34:56Z`).
- Status filters use lowercase names: `running`, `success`, `error`, or `unspecified`.
- The gateway is stateless; replicas can be added behind a load balancer if needed.
