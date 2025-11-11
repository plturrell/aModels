#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <host> <port> [timeout_seconds]

Waits until TCP <host>:<port> is accepting connections (default timeout 60s).
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

HOST="$1"
PORT="$2"
TIMEOUT="${3:-60}"

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Port must be numeric. Got: $PORT" >&2
  exit 2
fi

if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]]; then
  echo "Timeout must be numeric seconds. Got: $TIMEOUT" >&2
  exit 3
fi

DEADLINE=$((SECONDS + TIMEOUT))

while (( SECONDS <= DEADLINE )); do
  if exec 3<>"/dev/tcp/${HOST}/${PORT}" 2>/dev/null; then
    exec 3<&-
    exec 3>&-
    echo "Service ${HOST}:${PORT} is ready."
    exit 0
  fi
  sleep 1
done

echo "Timed out after ${TIMEOUT}s waiting for ${HOST}:${PORT}" >&2
exit 4
