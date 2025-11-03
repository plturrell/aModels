#!/usr/bin/env bash
# Submit a sample /graph payload using the sanitized SGMI fixtures.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_DIR=$(cd "${ROOT_DIR}/.." && pwd)/data/training_sgmi/pipeline_metamodel
LOG_DIR=$(cd "${ROOT_DIR}/.." && pwd)/logs/sgmi_pipeline

mkdir -p "${LOG_DIR}"

target_url=${1:-http://localhost:8081/graph}

json_table=$(python3 - <<'PY' "${DATA_DIR}/sgmi_table_pipeline.json"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)
hive_ddl=$(python3 - <<'PY' "${DATA_DIR}/sgmi_hive_pipeline.hql"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)
controlm_xml=$(python3 - <<'PY' "${DATA_DIR}/sgmi_controlm_pipeline.xml"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)

for path in "${json_table}" "${hive_ddl}" "${controlm_xml}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing pipeline metamodel artifact: ${path}" >&2
    exit 1
  fi
done

tmp_payload=$(mktemp)
tmp_response=$(mktemp)
trap 'rm -f "${tmp_payload}" "${tmp_response}"' EXIT

python3 - <<'PY' "${tmp_payload}" "${json_table}" "${hive_ddl}" "${controlm_xml}"
import json
import pathlib
import sys

payload_path = pathlib.Path(sys.argv[1])
json_table = pathlib.Path(sys.argv[2])
hive_ddl = pathlib.Path(sys.argv[3])
controlm_xml = pathlib.Path(sys.argv[4])

def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")

payload = {
    "json_tables": [str(json_table)],
    "hive_ddls": [read_text(hive_ddl)],
    "sql_queries": [
        "INSERT INTO sgmi_demo.target (order_id, amount) "
        "SELECT order_id, amount FROM sgmi_demo.source",
    ],
    "control_m_files": [str(controlm_xml)],
    "project_id": "sgmi-demo",
    "system_id": "sgmi-demo-system",
    "information_system_id": "sgmi-demo-info",
}

payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[$timestamp] Submitting SGMI pipeline metamodel payload to ${target_url}" >&2

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

if ! validation_summary=$(
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
); then
  validation_summary='{"http_status":0,"nodes":0,"edges":0,"controlm_jobs":0,"tables":0,"validation":"validation_script_error"}'
fi

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
    echo "Pipeline metamodel Postgres validation failed. See ${log_file} for details." >&2
    exit 1
  fi
fi

if [[ "${status_code}" -ne 200 || "${validation_state}" != "ok" ]]; then
  echo "Pipeline metamodel submission failed validation. See ${log_file} for details." >&2
  exit 1
fi
