#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_DIR="$ROOT_DIR/gateway"
export PYTHONPATH="$ROOT_DIR"
export POSTGRES_LANG_SERVICE_ADDR="${POSTGRES_LANG_SERVICE_ADDR:-localhost:50055}"
export POSTGRES_LANG_DB_DSN="${POSTGRES_LANG_DB_DSN:-postgres://user@localhost:5432/postgres?sslmode=disable}"
HOST="${FASTAPI_HOST:-0.0.0.0}"
PORT="${FASTAPI_PORT:-8000}"
RELOAD_FLAG="${FASTAPI_RELOAD:-false}"

if [[ ! -f "$GATEWAY_DIR/requirements.txt" ]]; then
  echo "[run_gateway] cannot find gateway directory" >&2
  exit 1
fi

cd "$GATEWAY_DIR"

if [[ -d .venv ]]; then
  source .venv/bin/activate
else
  echo "[run_gateway] creating local venv"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

CMD=(python -m uvicorn gateway.app:app --host "$HOST" --port "$PORT")
if [[ "$RELOAD_FLAG" =~ ^(1|true|TRUE)$ ]]; then
  CMD+=(--reload)
fi

exec "${CMD[@]}"
