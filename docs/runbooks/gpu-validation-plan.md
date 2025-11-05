# GPU Validation Playbook

Use this checklist once you have access to a Docker-enabled GPU host. It exercises the full runtime, verifies Goose migrations, and runs Deep Research regression tests end-to-end.

## Prerequisites

- Repo cloned at the latest `main` (`git pull origin main`).
- Large model assets in `models/` (fetch via release artifacts or Kaggle as described in `README.md` if you need LocalAI to respond).
- Python 3.11+ and Docker/Compose available on the GPU host.

## Steps

1. **Restore local Python environment**
   ```bash
   cd aModels
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e models/open_deep_research pytest-asyncio
   ```

2. **Run Deep Research regressions**
   ```bash
   python -m pytest models/open_deep_research/tests
   ```
   Ensures catalog tool registration logic still gates correctly.

3. **Start infrastructure stack**
   ```bash
   docker compose -f infrastructure/docker/compose.yml build
   docker compose -f infrastructure/docker/compose.yml up -d
   ```
   This spins up HANA, catalog, LocalAI, Deep Research, etc. Wait for services to report healthy (`docker compose ps`).

4. **Execute Goose migrations explicitly (optional but recommended)**
   ```bash
   docker compose exec catalog \
     go run ./cmd/migrate up

   docker compose exec extract-service \
     go run ./services/extract/cmd/migrate up

   go run ./cmd/migrate-graph status  # Uses SQLITE_MIGRATIONS_DSN / EXTRACT_SQLITE_PATH

   go run ./services/training/cmd/migrate up
   ```
   Confirms Neo4j + Postgres/SQLite migrations succeed across services.

5. **Sanity-check Deep Research endpoint**
   ```bash
   curl -X POST http://localhost:8085/research \
     -H 'Content-Type: application/json' \
     -d '{"query": "Summarise data lineage for customer analytics"}'
   ```
   You should receive a JSON payload with `status` and `report`.

6. **Verify research report persistence**
   ```bash
   psql postgres://postgres:postgres@localhost:5432/amodels?sslmode=disable \
     -c "SELECT count(*) FROM research_reports;"
   ```
   The count should increment after a successful Deep Research call.

7. **Clean up**
   ```bash
   docker compose -f infrastructure/docker/compose.yml down
   deactivate
   ```

Record results in your run log (success, issues encountered, service logs). If any migration or service fails, capture logs before tearing down (`docker compose logs <service>`).
