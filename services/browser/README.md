## Browser Service

This directory contains the browser integration components for the `aModels` repository:

- **Chromium Extension**: Browser extension for aModels integration
- **Shell UI**: React-based dashboard with analytics, SGMI explorer, and LocalAI chat
- **Open Canvas**: Advanced AI-powered document and code editor (LangChain)

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
- `SHELL_RUNTIME_ENDPOINT` – base URL for the runtime analytics service (defaults to `${SHELL_GATEWAY_URL}/runtime`).

The Go shell reverse-proxies `/dms/*` and `/agentflow/*` to the configured upstreams so the UI can call those services without CORS changes.

Quick verification once the shell is running:

```bash
curl -s http://localhost:4173/dms/documents | jq .
curl -s http://localhost:4173/agentflow/flows | jq .
curl -s http://localhost:4173/api/runtime/analytics/dashboard | jq .
```

These requests should return the same JSON payloads your FastAPI services expose on their native ports.

### Runtime analytics dashboard smoke test

1. Start the runtime service (see `services/runtime/README.md`).
2. Launch the shell with `SHELL_RUNTIME_ENDPOINT` pointing at the runtime service.
3. Open http://localhost:4173 in the browser and navigate to the analytics dashboard page; confirm charts and metrics render.
4. Alternatively, run `curl -s http://localhost:4173/api/runtime/analytics/dashboard | jq .` and ensure the response includes `stats` and `templates`.

## Open Canvas Integration

Open Canvas is an AI-powered document and code editor from LangChain that provides advanced collaboration features with LLMs. It has been integrated into the browser service to enhance the UI with:

- **Advanced Text Editor**: Rich markdown editing with live preview
- **Code Editor**: Multi-language support with syntax highlighting  
- **AI Collaboration**: Chat with AI to iteratively edit documents and code
- **Version History**: Track all changes over time
- **Memory System**: AI remembers your preferences across sessions
- **Quick Actions**: Pre-built and custom prompts for common tasks

### Quick Start

```bash
# Automated setup
cd /home/aModels/services/browser
./setup-opencanvas.sh

# Start Open Canvas
cd open-canvas
./start-all.sh
```

### Access Points

- **Open Canvas UI**: http://localhost:3000 (document/code editor)
- **Shell UI**: http://localhost:4173 (analytics/dashboards)
- **LangGraph API**: http://localhost:2024 (agent backend)

### Documentation

- **Quick Start Guide**: [QUICKSTART_OPENCANVAS.md](QUICKSTART_OPENCANVAS.md)
- **Integration Guide**: [OPEN_CANVAS_INTEGRATION.md](OPEN_CANVAS_INTEGRATION.md)
- **LocalAI Adapter**: [opencanvas-localai-adapter.ts](opencanvas-localai-adapter.ts)
- **Docker Compose**: [docker-compose.opencanvas.yml](docker-compose.opencanvas.yml)

### LocalAI Integration

Open Canvas is configured to use aModels' LocalAI instance instead of external APIs. Ensure LocalAI is running:

```bash
# Check LocalAI status
curl http://localhost:8080/v1/models

# Start LocalAI if needed
cd /home/aModels
make -f Makefile.services start-localai
```

### Features

1. **Document Writing**: Create and edit markdown documents with AI assistance
2. **Code Development**: Write and iterate on code in multiple languages
3. **Mixed Content**: Combine prose and code in the same document
4. **Version Control**: Track all versions and revert to previous states
5. **Personalization**: AI learns your style and preferences over time
6. **Custom Actions**: Create reusable prompts for frequent tasks

For detailed setup instructions and troubleshooting, see the documentation links above.

## Complete UI Stack

Beyond the core browser services, aModels includes additional UIs that can be integrated:

### Available UIs

| Service | Port | Status | Purpose |
|---------|------|--------|---------|
| Shell UI | 4173 | ✅ Integrated | Main dashboard |
| Open Canvas | 3000 | ✅ Integrated | AI document/code editor |
| LangFlow | 7860 | ⚠️ Available | Visual flow builder |
| Gitea | 3003 | ✅ Running | Git repository |
| Neo4j Browser | 7474 | ✅ Running | Graph database |
| Adminer | 8082 | ⚠️ Optional | PostgreSQL admin |
| Jaeger | 16686 | ⚠️ Optional | Distributed tracing |
| Grafana | 3001 | ⚠️ Optional | Metrics dashboards |

### Deploy Full Stack

```bash
# Deploy all UI services
cd /home/aModels/services/browser
docker-compose -f docker-compose.yml -f docker-compose.ui-stack.yml up -d

# Access service portal
open http://localhost:8888
```

### Documentation

- **[UI Services Assessment](UI_SERVICES_ASSESSMENT.md)** - Complete review and ratings (52/100)
- **[UI Stack Guide](START_UI_STACK.md)** - Starting and managing all UIs
- **[Build Guide](BUILD.md)** - Build instructions for all services

### Integration Status

**Overall Score: 52/100**

- ✅ **Core Browser** (90%): Shell UI + Open Canvas integrated
- ⚠️ **LangFlow** (40%): Available but not integrated
- ⚠️ **Observability** (20%): OpenLLMetry needs OTEL stack
- ✅ **Databases** (60%): Neo4j browser available, PostgreSQL UIs optional
- ⚠️ **Goose UI** (30%): Desktop-only, needs web port

See [UI_SERVICES_ASSESSMENT.md](UI_SERVICES_ASSESSMENT.md) for detailed analysis and roadmap.
