#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/relational_rt_example.yaml"
EVAL_CONFIG="${EVAL_CONFIG:-${ROOT_DIR}/configs/eval_financed_emission_regression.yaml}"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/main_schedule"
PRETRAIN_CKPT="${CHECKPOINT_DIR}/rt_pretrain.pt"
FINETUNE_CKPT="${CHECKPOINT_DIR}/rt_finetuned.pt"
ARC_DATA_DIR="${ARC_DATA_DIR:-${ROOT_DIR}/data/arc-agi}"
ARC2_DATA_DIR="${ARC2_DATA_DIR:-${ROOT_DIR}/data/arc-agi-2}"
ARC_MODEL="${ARC_MODEL:-hybrid}"
ARC_LIMIT="${ARC_LIMIT:-0}"
ARC_METRICS_DIR="${CHECKPOINT_DIR}/arc_metrics"

mkdir -p "${CHECKPOINT_DIR}" "${ARC_METRICS_DIR}"
timestamp="$(date +%Y%m%dT%H%M%S)"
python3 "${ROOT_DIR}/scripts/train_relational_transformer.py" \
  --config "${CONFIG}" \
  --mode pretrain \
  --auto-resume \
  --checkpoint-out "${PRETRAIN_CKPT}"
FT_ARGS=(--config "${CONFIG}" --mode fine-tune --checkpoint-out "${FINETUNE_CKPT}" --auto-resume)
if [[ ! -f "${FINETUNE_CKPT}" ]]; then
  FT_ARGS+=(--checkpoint-in "${PRETRAIN_CKPT}")
fi
python3 "${ROOT_DIR}/scripts/train_relational_transformer.py" "${FT_ARGS[@]}"

if [[ -f "${EVAL_CONFIG}" ]]; then
  python3 "${ROOT_DIR}/scripts/eval_relational_transformer.py" \
    --config "${CONFIG}" \
    --checkpoint "${PRETRAIN_CKPT}" \
    --eval-config "${EVAL_CONFIG}" \
    --output "${CHECKPOINT_DIR}/metrics_pretrain.json"

  python3 "${ROOT_DIR}/scripts/eval_relational_transformer.py" \
    --config "${CONFIG}" \
    --checkpoint "${FINETUNE_CKPT}" \
    --eval-config "${EVAL_CONFIG}" \
    --output "${CHECKPOINT_DIR}/metrics_finetune.json"
else
  echo "⚠️  Evaluation config not found at ${EVAL_CONFIG}; skipping nightly metrics." >&2
fi

run_arc_eval() {
  local dataset="$1"
  local label="$2"
  if [[ ! -d "${dataset}" && ! -f "${dataset}" ]]; then
    echo "⚠️  ARC dataset ${dataset} not found; skipping ${label} evaluation." >&2
    return
  fi
  local out_file="${ARC_METRICS_DIR}/${label}_${timestamp}.json"
  echo "▶️  Running ARC benchmark (${label}) against ${dataset}"
  go run -C "${ROOT_DIR}" ./cmd/aibench run \
    --task=arc \
    --model "${ARC_MODEL}" \
    --data "${dataset}" \
    --limit "${ARC_LIMIT}" \
    --out "${out_file}"
}

if command -v go >/dev/null 2>&1; then
  run_arc_eval "${ARC_DATA_DIR}" "arc_agi"
  run_arc_eval "${ARC2_DATA_DIR}" "arc_agi2"
else
  echo "⚠️  Go toolchain is unavailable; skipping ARC benchmarking." >&2
fi
