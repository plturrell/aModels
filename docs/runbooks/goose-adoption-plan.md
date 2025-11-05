# Goose Rollout Plan for Non-Catalog Services

Last updated: 2025-11-05

Catalog now uses Goose for both PostgreSQL and Neo4j migrations. The remaining services still initialise schema directly in code or via ad-hoc scripts, which leaves us exposed to drift. This plan outlines the concrete steps required to bring the rest of the estate in line.

## Target Services & Datastores

| Service  | Datastores in use | Current bootstrap approach | Required Goose work |
|----------|-------------------|----------------------------|---------------------|
| extract  | SQLite (`extract.db`), optional Postgres (vector store), Redis (cache), Neo4j (graph), filesystem exports | Schema created in Go initialisation paths; no versioning for SQLite/Neo4j | Introduce `services/extract/migrations/` for SQLite + Neo4j, add migration runner (`cmd/migrate`) and wire via `RUN_MIGRATIONS` |
| graph    | Neo4j, Redis (cache), optional Postgres | Neo4j schema seeded via code (constraints/indexes) | Mirror catalog Neo4j migration runner, add Goose-style `.cypher` files and migrate CLI |
| training | Postgres (metrics, routing), Redis (AB tests/cache) | SQL schema compiled into migration scripts in `pipeline.py` / helper modules | Create Postgres `migrations/` directory, generate initial Goose migration, add Make target/CLI runner |

## Execution Steps

1. **Baseline Analysis**
   - For each service, list all schema objects currently created on startup (tables, indexes, constraints).
   - Capture the canonical schema as SQL / Cypher files to form migration `0001_initial.*`.

2. **Scaffold Migration Runners**
   - Create `services/<service>/migrations/` with initial Goose files:
     - `0001_initial.sql` (`.cypher` for Neo4j).
   - Implement `services/<service>/cmd/migrate/main.go` mirroring the catalog runner:
     - `up`, `down`, `status`.
     - Support `RUN_MIGRATIONS=true` env var for auto-execution at service startup.

3. **Service Wiring**
   - Update each service’s main server boot to call the migration runner when `RUN_MIGRATIONS` is set (development/test only).
   - Document new env vars (`SQL_MIGRATIONS_DSN`, `NEO4J_URI`, etc.).

4. **CI Enforcement**
   - Extend the GitHub workflow to spin up service-specific datastores and execute:
     - `go run services/<service>/cmd/migrate up`.
     - `go run services/<service>/cmd/migrate down`.
   - Fail the build if migrations diverge or new migrations are missing.

5. **Documentation**
   - Update each service README with migration usage instructions.
   - Add links back to this runbook from `docs/architecture.md`.

## Open Questions / Follow-ups

- Redis structures (e.g., key prefixes) are largely ephemeral; document manual seeding rather than using Goose.
- Extract’s generated training exports need compatibility validation when SQLite schema evolves—plan regression tests before enabling Goose in production.
- Consider common tooling (shared migration CLI) if more services adopt Goose to avoid duplicated logic.

Once the work above lands, the only manual database initialisation will be legacy paths explicitly called out in service docs.

