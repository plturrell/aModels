# Langflow Integration Notes

## Pivot Overview

The Layer 4 Flow project is no longer attempting a like‑for‑like Go port of the Langflow backend. Instead, we now treat the upstream Python service as the source of truth and provide Go utilities that:

- synchronise our curated JSON flows into a running Langflow instance,
- execute flows remotely while keeping command-line ergonomics consistent with the rest of the agenticAiETH tooling,
- surface diagnostic helpers for request/response inspection so Go developers can continue to reason about payloads locally.

This pragmatic approach lets us iterate on flow content in Go while leveraging the mature Langflow runtime and UI.

## What Exists Today

- **HTTP client (`internal/langflow`)** – Minimal wrapper for `/api/v1/flows`, `/api/v1/flows/import`, `/api/v1/flows/{id}/run`, and `/api/v1/version`.
- **Catalog loader (`pkg/catalog`)** – Enumerates JSON definitions under `flows/`, parsing basic metadata and exposing fully raw payloads for import.
- **Runner (`runner`)** – Handles import orchestration (force/overwrite flags) and executes runs, returning the JSON response untouched.
- **CLI (`cmd/flow-run`)** – End-user entry point supporting:
  - `--probe` to confirm version handshake,
  - `--list` to inspect remote flows,
  - `--ensure`/`--force` to control synchronisation,
  - `--print-request` / `--print-response` for payload auditing,
  - registry-backed ID tracking so repeat runs can skip re-import.
- **Registry store (`store/langflow_registry.json`)** – JSON map from local catalog IDs to the Langflow IDs assigned during import.

## Operational Flow

1. Authors edit JSON specs in `flows/…`.
2. `flow-run --flow-id <id> --ensure` imports the spec and records the remote ID.
3. Subsequent `flow-run --flow-id <id>` calls reuse the cached ID (unless `--ensure` is set).
4. Optional `--print-request` lets developers compare payloads against Langflow’s OpenAPI schema without sending traffic.

## Recommended Practices

- Keep the registry file under source control ignore rules; treat it as deploy-specific state.
- Use `--force=false` in pre-production environments where Langflow flows are edited manually to avoid overwriting human changes.
- When Langflow responds with validation errors, re-run with `--print-request` to capture the exact payload that failed validation.
- Pair `--probe` with CI smoke checks to fail fast if the Langflow endpoint is unreachable.

## Future Enhancements

1. **Streaming support** – Extend the client to consume Langflow’s streaming endpoints once the upstream API stabilises.
2. **Structured typing** – Introduce typed wrappers for common response envelopes (success/error) to simplify downstream consumption.
3. **Selective sync** – Allow importing multiple flows via globbing or categories without having to list every ID manually.
4. **Observability hooks** – Emit structured logs when registry entries change, assisting auditors in tracking flow promotions.

The Go code remains intentionally small so that the team can continue deriving patterns from Python while relying on a single Langflow runtime. Any future work should preserve this separation: Go manages assets and automation, Langflow handles execution.
