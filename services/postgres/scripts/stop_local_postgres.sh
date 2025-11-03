#!/usr/bin/env bash

# Stops the PostgreSQL instance started by start_local_postgres.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="$ROOT_DIR/.build"
INSTALL_DIR="$BUILD_ROOT/postgres/installed"
DATA_DIR="${BUILD_ROOT}/pgdata"

if [[ ! -x "$INSTALL_DIR/bin/pg_ctl" || ! -d "$DATA_DIR" ]]; then
  echo "[stop_local_postgres] No local postgres instance appears to be installed." >&2
  exit 0
fi

if ! "$INSTALL_DIR/bin/pg_ctl" -D "$DATA_DIR" status >/dev/null 2>&1; then
  echo "[stop_local_postgres] Postgres is not running." >&2
  exit 0
fi

"$INSTALL_DIR/bin/pg_ctl" -D "$DATA_DIR" stop
echo "[stop_local_postgres] Postgres stopped."
