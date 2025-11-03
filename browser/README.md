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


