#!/usr/bin/env bash

# Applies all SQL migrations in chronological order against the configured Postgres DSN.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATIONS_DIR="$ROOT_DIR/migrations"
BUILD_ROOT="$ROOT_DIR/.build"
LOCAL_PSQL_BIN="$BUILD_ROOT/postgres/installed/bin/psql"
DEFAULT_DSN="postgres://user@localhost:5432/postgres?sslmode=disable"

POSTGRES_DSN="${POSTGRES_DSN:-$DEFAULT_DSN}"

log() {
  echo "[apply_migrations] $*"
}

if [[ ! -d "$MIGRATIONS_DIR" ]]; then
  log "No migrations directory found at $MIGRATIONS_DIR"
  exit 0
fi

PSQL_BIN="${PSQL_BIN:-}"
if [[ -n "$PSQL_BIN" && ! -x "$PSQL_BIN" ]]; then
  echo "[apply_migrations] Provided PSQL_BIN ($PSQL_BIN) is not executable" >&2
  exit 1
fi

if [[ -z "$PSQL_BIN" ]]; then
  if [[ -x "$LOCAL_PSQL_BIN" ]]; then
    PSQL_BIN="$LOCAL_PSQL_BIN"
  elif command -v psql >/dev/null 2>&1; then
    PSQL_BIN="$(command -v psql)"
  else
    echo "[apply_migrations] psql binary not found; install Postgres client tools or set PSQL_BIN" >&2
    exit 1
  fi
fi

shopt -s nullglob
migration_files=("$MIGRATIONS_DIR"/*.sql)
shopt -u nullglob

if (( ${#migration_files[@]} == 0 )); then
  log "No SQL migration files discovered in $MIGRATIONS_DIR"
  exit 0
fi

log "Applying migrations using $PSQL_BIN"
for migration in "${migration_files[@]}"; do
  log "Applying $(basename "$migration")"
  "$PSQL_BIN" "$POSTGRES_DSN" -f "$migration" >/dev/null
done

log "Migrations complete"
