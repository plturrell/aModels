# aModels Browser Shell

This directory contains an Electron wrapper that pairs a Chromium browsing surface with a React control panel for gateway actions, LocalAI chat, and telemetry. The left pane defaults to a repository-backed home page (`home/index.html`) and can navigate anywhere you choose, while the right pane hosts the Vite-built UI.

## Quick start

```bash
cd services/browser/shell
npm install
npm start        # builds the React panel (Vite) and launches Electron
```

The shell launches a single window split into:

- **Left panel** – Full Chromium browsing surface (`BrowserView`). It defaults to the bundled home page (`home/index.html`), but you can override it via the `AMODELS_SHELL_START_URL` environment variable.
- **Right panel** – The React control panel compiled by Vite (`ui/`). It listens to navigation events via `shellBridge.onNavigation`, mirrors the active tab metadata, and exposes gateway actions, LocalAI chat, and a structured log.

## Notes

- `npm start` always runs `vite build` to ensure `ui/dist/` exists. For rapid iteration, you can run `npm run build:ui -- --watch` in one terminal and `electron .` in another.
- The bundled React UI talks directly to gateway endpoints via fetch; update `ui/src/App.jsx` if you need additional actions.
- Browsing state persists in Electron’s default profile (`app.getPath('userData')`). Delete that directory to reset session data.
- Use `Cmd/Ctrl+K` to open the command palette and trigger navigation, gateway calls, or benchmarks without leaving the keyboard.
- Toggle light/dark appearance from the toolbar; the shell remembers your preference between sessions.
- Regenerate the SGMI lineage dataset with `./scripts/generate_sgmi_flow.py` if the source files change.

## Next steps

- Stream benchmark history to persistent storage (e.g., sqlite) if you need to compare runs across sessions.
- Feed additional telemetry endpoints (training jobs, LocalAI stats) into the dashboard for richer observability.
- Integrate the legacy Chromium extension codepaths only when Chrome-based workflows are required; the shell no longer depends on the popup.
