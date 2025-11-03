#!/usr/bin/env bash
# Ingest the full SGMI dataset (tables + views + Control-M) and emit view lineage metadata.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${ROOT_DIR}/.." && pwd)
DATA_ROOT="${REPO_ROOT}/data/training_sgmi"
LOG_DIR="${REPO_ROOT}/logs/sgmi_pipeline"

mkdir -p "${LOG_DIR}"

target_url=${1:-http://localhost:19080/graph}

json_tables=(
  "${DATA_ROOT}/JSON_with_changes.json"
)

hive_ddls=(
  "${DATA_ROOT}/HIVE DDLS/sgmisit_all_tables_statement.hql"
  "${DATA_ROOT}/HIVE DDLS/sgmisitetl_all_tables_statement.hql"
  "${DATA_ROOT}/HIVE DDLS/sgmisitstg_all_tables_statement.hql"
  "${DATA_ROOT}/HIVE DDLS/sgmisit_view.hql"
)

controlm_files=(
  "${DATA_ROOT}/SGMI-controlm/catalyst migration prod 640.xml"
)

default_view_store="${REPO_ROOT}/agenticAiETH_layer4_AgentFlow/store"
view_registry_out=${SGMI_VIEW_REGISTRY_OUT:-${default_view_store}/sgmi_view_lineage.json}
view_summary_out=${SGMI_VIEW_SUMMARY_OUT:-$(dirname "${view_registry_out}")/sgmi_view_summary.json}

missing=0
for path in "${json_tables[@]}" "${hive_ddls[@]}" "${controlm_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] required artifact not found: ${path}" >&2
    missing=1
  fi
done
if [[ ${missing} -ne 0 ]]; then
  exit 1
fi

tmp_payload=$(mktemp)
tmp_response=$(mktemp)
view_tmp_dir=$(mktemp -d)
trap 'rm -f "${tmp_payload}" "${tmp_response}"; rm -rf "${view_tmp_dir}"' EXIT

join_paths() {
  local IFS=":"
  echo "$*"
}

export SGMI_JSON_FILES=$(join_paths "${json_tables[@]}")
export SGMI_DDL_FILES=$(join_paths "${hive_ddls[@]}")
export SGMI_CONTROLM_FILES=$(join_paths "${controlm_files[@]}")
export SGMI_VIEW_TMP_DIR="${view_tmp_dir}"
export SGMI_VIEW_REGISTRY_OUT="${view_registry_out}"
export SGMI_VIEW_SUMMARY_OUT="${view_summary_out}"

python3 "${SCRIPT_DIR}/sgmi_view_builder.py" "${tmp_payload}"

timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[${timestamp}] Submitting SGMI full payload to ${target_url}" >&2

set +e
http_status=$(curl -sSL -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  --data-binary "@${tmp_payload}" \
  -o "${tmp_response}" \
  "${target_url}")
curl_exit=$?
set -e

if [[ ${curl_exit} -ne 0 ]]; then
  echo "curl failed with exit code ${curl_exit}" >&2
  http_status="000"
fi

validation_summary=$(
  python3 - "${tmp_response}" "${http_status}" <<'PY'
import json
import pathlib
import sys

resp_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None

try:
    status = int(sys.argv[2])
except (ValueError, IndexError):
    status = 0

result = {
    "http_status": status,
    "nodes": 0,
    "edges": 0,
    "controlm_jobs": 0,
    "tables": 0,
    "validation": "ok",
}

try:
    if status != 200:
        result["validation"] = "http_error"
        raise SystemExit

    if resp_path is None or not resp_path.exists():
        result["validation"] = "missing_response"
        raise SystemExit

    payload = json.loads(resp_path.read_text(encoding="utf-8"))

    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []
    result["nodes"] = len(nodes)
    result["edges"] = len(edges)
    result["controlm_jobs"] = sum(1 for n in nodes if str(n.get("type")).lower() == "control-m-job")
    result["tables"] = sum(1 for n in nodes if str(n.get("type")).lower() == "table")

    if result["nodes"] == 0 or result["edges"] == 0:
        result["validation"] = "empty_graph"
    elif result["controlm_jobs"] == 0:
        result["validation"] = "missing_controlm"
    elif result["tables"] == 0:
        result["validation"] = "missing_tables"
except json.JSONDecodeError:
    result["validation"] = "invalid_json_response"
except SystemExit:
    pass
except Exception as exc:
    result["validation"] = "validation_exception"
    result["error"] = str(exc)

print(json.dumps(result))
PY
)

log_file="${LOG_DIR}/sgmi_pipeline_$(date -u +%Y%m%d).log"
echo "${timestamp} ${validation_summary}" >> "${log_file}"
echo "${validation_summary}"

status_code=$(printf '%s' "${validation_summary}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["http_status"])')
validation_state=$(printf '%s' "${validation_summary}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["validation"])')

if [[ "${status_code}" -eq 200 && "${validation_state}" == "ok" && -n "${POSTGRES_CATALOG_DSN:-}" ]]; then
  echo "Validating Postgres replication…" >&2
  if ! db_validation=$(
    cd "${ROOT_DIR}" && go run ./cmd/extract-validate -timeout 5s
  ); then
    echo "Postgres validation failed" >&2
    echo "${timestamp} ${db_validation}" >> "${log_file}"
    exit 1
  fi
  echo "${db_validation}"
  echo "${timestamp} ${db_validation}" >> "${log_file}"
  db_status=$(printf '%s' "${db_validation}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  if [[ "${db_status}" != "ok" ]]; then
    echo "SGMI Postgres validation failed. See ${log_file} for details." >&2
    exit 1
  fi
fi

if [[ "${status_code}" -ne 200 || "${validation_state}" != "ok" ]]; then
  echo "SGMI submission failed validation. See ${log_file} for details." >&2
  exit 1
fi
