#!/usr/bin/env bash
# Launch the Extract HTTP/GRPC service after verifying dependencies.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXTRACT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${EXTRACT_ROOT}/.." && pwd)
LOG_DIR="${REPO_ROOT}/logs/extract"

mkdir -p "${LOG_DIR}"

if ! command -v go >/dev/null 2>&1; then
  echo "go toolchain is required but not found in PATH" >&2
  exit 2
fi

if [[ ! -x "${SCRIPT_DIR}/wait_for_service.sh" ]]; then
  echo "missing helper script: ${SCRIPT_DIR}/wait_for_service.sh" >&2
  exit 3
fi

POSTGRES_DSN=${POSTGRES_CATALOG_DSN:-}
if [[ -n "${POSTGRES_DSN}" ]]; then
  host_port=$(python3 - "${POSTGRES_DSN}" <<'PY'
import sys
from urllib.parse import urlparse

dsn = sys.argv[1].strip()
url = urlparse(dsn)
if not url.hostname:
    raise SystemExit("unable to parse host from DSN")
port = url.port or 5432
print(f"{url.hostname}:{port}")
PY
  ) || {
    echo "failed to parse POSTGRES_CATALOG_DSN for wait check" >&2
    exit 4
  }

  host=${host_port%:*}
  port=${host_port##*:}
  timeout=${WAIT_FOR_POSTGRES_TIMEOUT:-60}
  echo "Waiting for Postgres at ${host}:${port} (timeout ${timeout}s)…"
  "${SCRIPT_DIR}/wait_for_service.sh" "${host}" "${port}" "${timeout}"
else
  echo "Warning: POSTGRES_CATALOG_DSN not set; skipping Postgres availability check." >&2
fi

timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
log_file="${LOG_DIR}/extract_${timestamp}.log"

echo "Starting Extract service (logs: ${log_file})"

cd "${EXTRACT_ROOT}"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL go run . "$@" 2>&1 | tee -a "${log_file}" &
else
  echo "stdbuf not found; logs may be buffered" >&2
  go run . "$@" 2>&1 | tee -a "${log_file}" &
fi
child_pid=$!

trap 'echo "Stopping Extract service…"; kill ${child_pid} >/dev/null 2>&1 || true; wait ${child_pid} 2>/dev/null' INT TERM

wait ${child_pid}
exit_code=$?
echo "Extract service exited with status ${exit_code}"
exit ${exit_code}
