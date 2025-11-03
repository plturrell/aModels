#!/usr/bin/env bash

# Blocks until the Postgres Lang gRPC service passes HealthCheck.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO_BIN="${GO:-go}"
ADDR="${POSTGRES_LANG_SERVICE_ADDR:-localhost:50055}"
WAIT="${HEALTHCHECK_WAIT:-15s}"
TIMEOUT="${HEALTHCHECK_TIMEOUT:-2s}"

cd "$ROOT_DIR"
GOWORK=off "$GO_BIN" run ./cmd/healthcheck -addr="$ADDR" -wait="$WAIT" -timeout="$TIMEOUT"
