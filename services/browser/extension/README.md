## aModels Chromium Extension (MV3, minimal)

This is a minimal Manifest V3 extension scaffold. You can load it unpacked for quick testing, then evolve it into a full Vite/TypeScript/React build.

### Load Unpacked

1. Go to `chrome://extensions` and enable Developer mode.
2. Click "Load unpacked" and select this `browser/extension` directory.
3. Pin the extension from the toolbar if desired.

### What’s included

- `manifest.json` – Minimal MV3 manifest with action and placeholder entries.

### Upgrade path

- Add a build system (Vite + React/Preact) under `src/`.
- Update `manifest.json` to reference the built `dist` assets as `action.default_popup`, `service_worker`, etc.
- Implement UI that calls your Layer4 gateway (FastAPI) via a single base URL.


