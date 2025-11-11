## Chromium browser integration

This directory contains the Chromium extension and related docs for the `aModels` repository.

### Quick start: load the extension (unpacked)

1. Open Chromium/Chrome and navigate to `chrome://extensions`.
2. Enable Developer mode (top-right toggle).
3. Click "Load unpacked" and select the `browser/extension` folder in this repository.
4. The extension will appear in your extensions list. Pin it for quick access.

### Manifest V3 minimal skeleton

The `browser/extension` folder includes a minimal Manifest V3 `manifest.json` that you can extend. For a production build (Vite/TypeScript/React), add a build step and point the manifest to the built assets.

### Gateway settings

If you are running the Layer4 gateway locally (FastAPI) or remotely, expose a single base URL and update your extension code to call the gateway endpoints. Typical endpoints:

- `/healthz` – health checks for all services
- `/agentflow/run` – run an AgentFlow
- `/extract/ocr`, `/extract/schema-replication` – extract endpoints
- `/data/sql`, `/telemetry/recent` – Postgres/data endpoints

Recommended: store the gateway base URL in extension storage (options page) or an environment-specific config and read it at runtime.

### Next steps

- Add a build system (e.g., Vite + React/Preact + TypeScript) under `browser/extension/src` and update the manifest to include the built assets.
- Implement UI tabs (Extract, Graph, Flow, Data, Discover, Training, LocalAI, AgentFlow) that call the gateway.
- Ensure CORS is enabled on the gateway for extension origins.

### Shell UI (Chromium host)

`services/browser/shell/ui` contains the Perplexity-style host that now ships with live LocalAI chat, SGMI data exploration, and telemetry dashboards.

```bash
# install dependencies once
cd services/browser/shell/ui
npm install

# iterate locally
npm run dev

# or produce an embedded bundle for the Go server
npm run build

# regenerate dist assets and run the embedded server
cd ../../../..
make shell-serve SHELL_DMS_ENDPOINT=http://localhost:8084 SHELL_AGENTFLOW_ENDPOINT=http://localhost:8086
```

The embedded server automatically discovers gateway endpoints and falls back to repository artefacts when an API is unavailable.

Environment overrides:

- `SHELL_GATEWAY_URL` – base URL for the gateway (`GATEWAY_URL` / `http://localhost:8000` by default).
- `SHELL_LOCALAI_URL` – point to a LocalAI/OpenAI-compatible base (defaults to `LOCALAI_URL`).
- `SHELL_SGMI_JSON` – alternate path to a `json_with_changes.json` payload.
- `SHELL_MODELS_DIR` – override the repository `models/` folder root.
- `SHELL_SGMI_ENDPOINT` – overrides the inferred `SHELL_GATEWAY_URL/shell/sgmi/raw` endpoint (falls back to file on disk if unset/unavailable).
- `SHELL_TRAINING_DATA_ENDPOINT` – overrides the inferred `SHELL_GATEWAY_URL/shell/training/dataset` endpoint (falls back to derived summaries on failure).
- `SHELL_DMS_ENDPOINT` – target for Document Management Service requests (defaults to `${SHELL_GATEWAY_URL}/dms`).
- `SHELL_AGENTFLOW_ENDPOINT` – target for AgentFlow/LangFlow requests (defaults to `${SHELL_GATEWAY_URL}/agentflow`).

The Go shell reverse-proxies `/dms/*` and `/agentflow/*` to the configured upstreams so the UI can call those services without CORS changes.

Quick verification once the shell is running:

```bash
curl -s http://localhost:4173/dms/documents | jq .
curl -s http://localhost:4173/agentflow/flows | jq .
```

These requests should return the same JSON payloads your FastAPI services expose on their native ports.
