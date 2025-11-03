#!/usr/bin/env bash

# Starts a local PostgreSQL instance compiled from third_party/postgres.
# The first run will configure, build, and install Postgres into
# agenticAiETH_layer4_Postgres/.build/postgres/installed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="$ROOT_DIR/.build"
SRC_ROOT="$ROOT_DIR/../third_party/postgres"
WORKTREE_DIR="$BUILD_ROOT/postgres-src"
INSTALL_DIR="$BUILD_ROOT/postgres/installed"
DATA_DIR="${BUILD_ROOT}/pgdata"
LOG_FILE="${BUILD_ROOT}/postgres.log"

CPU_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

msg() {
  echo "[start_local_postgres] $*"
}

ensure_source_tree() {
  if [[ ! -d "$SRC_ROOT" ]]; then
    echo "Postgres source not found at $SRC_ROOT" >&2
    exit 1
  fi

  if [[ ! -d "$WORKTREE_DIR" ]]; then
    msg "Copying postgres source into ${WORKTREE_DIR}"
    mkdir -p "$(dirname "$WORKTREE_DIR")"

    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "$SRC_ROOT/" "$WORKTREE_DIR/"
    else
      cp -R "$SRC_ROOT" "$WORKTREE_DIR"
    fi
  fi
}

ensure_compiled_postgres() {
  if [[ -x "$INSTALL_DIR/bin/postgres" ]]; then
    return
  fi

  ensure_source_tree

  msg "Configuring postgres (without ICU support)"
  pushd "$WORKTREE_DIR" >/dev/null
  ./configure --prefix="$INSTALL_DIR" --without-icu

  msg "Building postgres (this can take a minute)"
  make -j"$CPU_COUNT"

  msg "Installing postgres into $INSTALL_DIR"
  make install
  popd >/dev/null
}

ensure_data_directory() {
  if [[ -d "$DATA_DIR/base" ]]; then
    return
  fi

  msg "Initialising data directory at $DATA_DIR"
  mkdir -p "$DATA_DIR"
  "$INSTALL_DIR/bin/initdb" -D "$DATA_DIR"
}

start_postgres() {
  if "$INSTALL_DIR/bin/pg_ctl" -D "$DATA_DIR" status >/dev/null 2>&1; then
    msg "Postgres already running"
    return
  fi

  msg "Starting postgres; logs => $LOG_FILE"
  "$INSTALL_DIR/bin/pg_ctl" -D "$DATA_DIR" -l "$LOG_FILE" start
}

ensure_compiled_postgres
ensure_data_directory
start_postgres

msg "Postgres ready (connection: postgres://user@localhost:5432/postgres?sslmode=disable)"
