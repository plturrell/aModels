#!/usr/bin/env bash

# Runs the Go unit tests (with go.work disabled) and optional gateway checks.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[run_tests] running Go tests (GOWORK=off)"
cd "$ROOT_DIR"
GOWORK=off go test ./...

if [[ -d gateway ]]; then
  echo "[run_tests] running gateway import check"
  pushd gateway >/dev/null
  if [[ -d .venv ]]; then
    source .venv/bin/activate
  fi
  PYTHONPATH="$ROOT_DIR" python -m compileall .
  popd >/dev/null
fi

echo "[run_tests] all tests completed"
