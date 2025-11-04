#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SCRIPT="${ROOT_DIR}/scripts/wait_for_service.sh"

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
wait_for_endpoint "${WAIT_FOR_REDIS:-}" 60
wait_for_endpoint "${WAIT_FOR_NEO4J:-}" 60
wait_for_endpoint "${WAIT_FOR_EXTRACT:-}" 60

CHAINRUNNER_BIN="${ROOT_DIR}/cmd/chainrunner/chainrunner"

if [[ -x "${CHAINRUNNER_BIN}" ]]; then
  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    exec "${CHAINRUNNER_BIN}" ${EXTRA_ARGS}
  else
    exec "${CHAINRUNNER_BIN}"
  fi
fi

cd "${ROOT_DIR}"
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  exec go run ./cmd/chainrunner ${EXTRA_ARGS}
else
  exec go run ./cmd/chainrunner
fi
