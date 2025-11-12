#!/usr/bin/env bash
set -euo pipefail

# Optional environment variables:
#   WAIT_FOR_POSTGRES=host:port[:timeout]
#   WAIT_FOR_HANA=host:port[:timeout]
#   WAIT_FOR_REDIS=host:port[:timeout]
#   WAIT_FOR_NEO4J=host:port[:timeout]
#   EXTRA_ARGS="..."  (additional flags for the extract binary)
#
# Example:
#   WAIT_FOR_POSTGRES=postgres:5432 WAIT_FOR_REDIS=redis:6379 ./scripts/start_extract.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SCRIPT="${ROOT_DIR}/scripts/wait_for_service.sh"
SERVER_BIN="${ROOT_DIR}/server"

if [[ ! -x "${WAIT_SCRIPT}" ]]; then
  echo "wait_for_service.sh not found or not executable at ${WAIT_SCRIPT}" >&2
  exit 1
fi

wait_for_endpoint() {
  local spec="$1"
  local default_timeout="$2"
  [[ -z "${spec}" ]] && return 0

  local host port timeout
  IFS=':' read -r host port timeout <<<"${spec}"
  if [[ -z "${timeout}" ]]; then
    timeout="${default_timeout}"
  fi

  echo "Waiting for ${host}:${port} (timeout ${timeout}s)..."
  "${WAIT_SCRIPT}" "${host}" "${port}" "${timeout}"
}

wait_for_endpoint "${WAIT_FOR_POSTGRES:-}" 90
wait_for_endpoint "${WAIT_FOR_HANA:-}" 120
wait_for_endpoint "${WAIT_FOR_REDIS:-}" 60
wait_for_endpoint "${WAIT_FOR_NEO4J:-}" 60

if [[ ! -x "${SERVER_BIN}" ]]; then
  echo "Extract server binary not found at ${SERVER_BIN}; did you run go build?" >&2
  exit 2
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  exec "${SERVER_BIN}" ${EXTRA_ARGS}
else
  exec "${SERVER_BIN}"
fi
