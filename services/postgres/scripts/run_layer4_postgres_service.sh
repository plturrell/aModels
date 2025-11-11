#!/usr/bin/env bash

# Runs the Postgres Layer4 gRPC service with a local DSN.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="$ROOT_DIR"
DEFAULT_DSN="postgres://user@localhost:5432/postgres?sslmode=disable"

POSTGRES_DSN="${POSTGRES_DSN:-$DEFAULT_DSN}"
export GRPC_PORT="${GRPC_PORT:-50055}"
AUTO_SETUP_POSTGRES="${AUTO_SETUP_POSTGRES:-true}"
AUTO_APPLY_MIGRATIONS="${AUTO_APPLY_MIGRATIONS:-true}"

is_truthy() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

if is_truthy "$AUTO_SETUP_POSTGRES" && [[ "$POSTGRES_DSN" == "$DEFAULT_DSN" ]]; then
  "$ROOT_DIR/scripts/start_local_postgres.sh"
fi

if is_truthy "$AUTO_APPLY_MIGRATIONS"; then
  POSTGRES_DSN="$POSTGRES_DSN" "$ROOT_DIR/scripts/apply_migrations.sh"
fi

echo "[run_layer4_postgres_service] POSTGRES_DSN=${POSTGRES_DSN}"
cd "$SERVICE_DIR"
GOWORK=off POSTGRES_DSN="$POSTGRES_DSN" GRPC_PORT="$GRPC_PORT" go run ./cmd/server/main.go
