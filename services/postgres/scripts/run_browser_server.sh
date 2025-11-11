#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BROWSER_DIR="$ROOT_DIR/agenticAiETH_layer4_Browser"

cd "$BROWSER_DIR"

is_truthy() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

AUTO_WIRE_POSTGRES=${AUTO_WIRE_POSTGRES:-true}
WAIT_FOR_POSTGRES=${WAIT_FOR_POSTGRES:-true}
POSTGRES_LANG_GATEWAY_URL=${POSTGRES_LANG_GATEWAY_URL:-http://localhost:8000}
POSTGRES_LANG_SERVICE_ADDR=${POSTGRES_LANG_SERVICE_ADDR:-localhost:50055}

args=("$@")

has_flag() {
  local flag="$1"
  for value in "${args[@]}"; do
    if [[ "${value}" == "$flag" || "${value}" == $flag=* ]]; then
      return 0
    fi
  done
  return 1
}

append_flag() {
  local flag="$1"
  local value="$2"
  if has_flag "$flag"; then
    return
  fi
  if [[ -n "$value" ]]; then
    args+=("$flag" "$value")
  fi
}

if [[ ! -x bin/browser-server ]]; then
  echo "[run_browser_server] building browser server binary"
  go build -o bin/browser-server ./cmd/browser-server
fi

if is_truthy "$AUTO_WIRE_POSTGRES"; then
  append_flag "--db-gateway-url" "$POSTGRES_LANG_GATEWAY_URL"
  append_flag "--telemetry-gateway-url" "$POSTGRES_LANG_GATEWAY_URL"
fi

args+=("--db-ui-only")

if is_truthy "$WAIT_FOR_POSTGRES"; then
  WAIT_SCRIPT="$ROOT_DIR/scripts/wait_for_service.sh"
  if [[ -x "$WAIT_SCRIPT" ]]; then
    POSTGRES_LANG_SERVICE_ADDR="$POSTGRES_LANG_SERVICE_ADDR" "$WAIT_SCRIPT"
  fi
fi

exec ./bin/browser-server "${args[@]}"
