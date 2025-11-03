#!/usr/bin/env bash
#
# Stop the LocalAI stack started by start_localai_stack.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${LOG_DIR}/localai_stack.pids"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PID file found (${PID_FILE}). Stack might already be stopped." >&2
  exit 0
fi

while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  pid="${line%%:*}"
  log="${line#*:}"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping PID ${pid}"
    kill "${pid}" 2>/dev/null || true
    for _ in {1..10}; do
      if kill -0 "${pid}" 2>/dev/null; then
        sleep 1
      else
        break
      fi
    done
    if kill -0 "${pid}" 2>/dev/null; then
      echo "  -> force killing ${pid}"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
  echo "  logs: ${log}"
done < "${PID_FILE}"

rm -f "${PID_FILE}"
echo "Stack stopped."
