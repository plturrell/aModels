#!/usr/bin/env bash
# Validate required environment variables for the Extract service.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: check_extract_env.sh <env_file>

Validates that required keys are present/properly formatted for Extract.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

ENV_FILE=$1
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 2
fi

required_vars=(
  POSTGRES_CATALOG_DSN
  LANGEXTRACT_API_URL
)

optional_vars=(
  POSTGRES_LANG_SERVICE_ADDR
  GLEAN_EXPORT_DIR
  REDIS_ADDR
)

while IFS= read -r line; do
  [[ -z "${line}" || "${line}" =~ ^# ]] && continue
  key=${line%%=*}
  value=${line#*=}
  export "$key"="${value}"
done <"${ENV_FILE}"

missing=0
for key in "${required_vars[@]}"; do
  if [[ -z "${!key-}" ]]; then
    echo "[ERROR] Required variable ${key} is missing or empty" >&2
    missing=1
  fi
done

if [[ ${missing} -ne 0 ]]; then
  exit 3
fi

printf '[OK] %s contains required configuration\n' "${ENV_FILE}"

for key in "${optional_vars[@]}"; do
  [[ -n "${!key-}" ]] || echo "[WARN] Optional variable ${key} is unset"
done
