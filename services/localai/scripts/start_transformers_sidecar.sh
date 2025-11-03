#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_transformers"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements-transformers.txt"
HOST=${TRANSFORMERS_CPU_HOST:-127.0.0.1}
PORT=${TRANSFORMERS_CPU_PORT:-8081}

echo "==> Starting transformers sidecar on ${HOST}:${PORT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "==> Installing/upgrading dependencies"
pip install --upgrade pip >/dev/null
pip install --upgrade -r "${REQUIREMENTS_FILE}" >/dev/null

cd "${ROOT_DIR}"

exec uvicorn services.transformers_cpu_server:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --no-access-log
