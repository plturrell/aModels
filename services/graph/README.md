# LangGraph-Go Port

## Overview

This directory hosts the Go-native port of the LangGraph monorepo. The layout mirrors the Python/TypeScript libraries to allow for incremental migration and feature parity with the existing ecosystem. The goal is to provide a high-performance, Go-native implementation of LangGraph that integrates seamlessly with the agenticAiETH project.

## Dependencies

- **Go:** Version 1.20 or higher

## Getting Started

To get started with the LangGraph-Go port, follow these steps:

1.  **Install Dependencies:** Ensure you have Go installed on your system.
2.  **Build the CLI:**

    ```bash
    go build ./cmd/langgraph
    ```

3.  **Run the Demo:**

    ```bash
    ./langgraph demo -input 3 -checkpoint sqlite:langgraph.dev.db -mode sync
    ```

## Modules

- `langgraph`: core execution engine, channel primitives, checkpoint APIs, and
  higher-level prebuilt agents.

The Go workspace (`go.work`) also pulls in the `agenticAiETH` projects that must
be integrated with the LangGraph runtime:

- `agenticAiETH_layer1_Blockchain` (maths infrastructure)
- `agenticAiETH_layer4_HANA` (enterprise orchestration)

## Next Steps

- [ ] Define the core graph data structures and execution loop (`pkg/graph`).
- [ ] Port channel semantics and streaming interfaces (`pkg/channels`).
- [ ] Model checkpoint contracts and implement at least one backend.
- [ ] Bring over the CLI surface and SDK entrypoints.
- [ ] Wrap the existing agenticAiETH maths/HANA components with LangGraph-native
   adapters so both ecosystems interoperate.

## Usage

### CLI Demo

You can exercise the initial Go runtime via the demo command:

```bash
go run ./cmd/langgraph demo -input 3 -checkpoint sqlite:langgraph.dev.db -mode sync
```

SQLite is the default for local development. Switch to `-checkpoint redis://localhost:6379/0` for a Redis-backed cache
or `-checkpoint hana` (with `HANA_*` environment variables set and the binary built with `-tags hana`) to persist
checkpoints in SAP HANA. Pass `-resume` to reuse previously saved state.

### External Service Integration

Set `AGENTSDK_FLIGHT_ADDR` (or export it in the environment where the graph server runs) to enrich every invocation with the live suite/tool catalog exposed by the Layer4 Agent SDK. The `/run` endpoint automatically injects `agent_catalog` and `agent_tools` into the initial state so downstream nodes can route work based on the currently attached MCP services. The server also exposes a convenience endpoint:

```bash
curl http://localhost:8081/agent/catalog
```

which proxies the Arrow Flight dataset as JSON for dashboards or health checks.

#### Extract Service

The ingestion workflow can now talk to the extract layer over both HTTP and gRPC, and it can mirror the latest `/graph` output over Arrow Flight:

- `EXTRACT_SERVICE_URL` – legacy REST endpoint (`http://extract-service:8081` by default).
- `EXTRACT_GRPC_ADDR` – optional gRPC endpoint (`host:9090`). When provided, the workflow prefers gRPC responses for entity extraction.
- `EXTRACT_FLIGHT_ADDR` – optional Flight endpoint (`host:8815`). The graph server exposes the cached nodes/edges at `/extract/graph` when set.

#### Postgres Lang Service

To inspect the lang operation telemetry in Postgres without leaving the graph service, provide:

- `POSTGRES_GRPC_ADDR` – gRPC endpoint for `PostgresLangService` (used by `/postgres/analytics` and `/postgres/operations/grpc`).
- `POSTGRES_FLIGHT_ADDR` – Arrow Flight endpoint for bulk operation retrieval (surfaced at `/postgres/operations`).

### HANA Configuration

- Copy `.env.hana.example` to `.env.hana` (ignored) and adjust credentials if needed. The sample values match the current development tenant.
- Export the variables before running Go tooling, e.g.: `export $(grep -v '^#' .env.hana | xargs)`.
- Alternatively, configure the environment in your shell profile or CI secrets manager.

## Testing

To run the tests for this component, including the live integration tests for HANA, use the following command:

```bash
go test -tags hana ./pkg/integration/hana -run Live -cover -count=1
```

The live test is skipped automatically when the environment is not present.

## Contributing

We welcome contributions to the LangGraph-Go port. Please follow these guidelines:

1.  **Code Style:** Adhere to the standard Go formatting and linting practices.
2.  **Testing:** All new features and bug fixes must be accompanied by unit tests.
3.  **Pull Requests:** Create a pull request with a clear description of your changes.
