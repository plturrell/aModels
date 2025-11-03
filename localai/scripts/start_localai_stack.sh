#!/usr/bin/env bash
#
# Start the full LocalAI stack:
#   * llama.cpp server for Phi 3.5 Mini (port 8081)
#   * llama.cpp server for Granite 4.0 hybrid (port 8082)
#   * vaultgemma LocalAI router (port 8080)
# Processes are backgrounded with stdout/stderr logged under logs/.
# A PID file (logs/localai_stack.pids) is written so stop_localai_stack.sh
# can terminate everything cleanly.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LLAMA_ROOT="${ROOT_DIR}/../third_party/llama.cpp"
BIN_DIR="${ROOT_DIR}/bin"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${LOG_DIR}/localai_stack.pids"

PHI_MODEL_QUANT_DEFAULT="${ROOT_DIR}/../agenticAiETH_layer4_Models/phi/phi-3-pytorch-phi-3.5-mini-instruct-v2-q4_0.gguf"
PHI_MODEL_F32_DEFAULT="${ROOT_DIR}/../agenticAiETH_layer4_Models/phi/phi-3.5-mini.gguf"
if [[ -n "${PHI_MODEL_PATH:-}" ]]; then
  PHI_MODEL="${PHI_MODEL_PATH}"
elif [[ -f "${PHI_MODEL_QUANT_DEFAULT}" ]]; then
  PHI_MODEL="${PHI_MODEL_QUANT_DEFAULT}"
else
  PHI_MODEL="${PHI_MODEL_F32_DEFAULT}"
fi
GRANITE_MODEL_QUANT_DEFAULT="${ROOT_DIR}/../agenticAiETH_layer4_Models/granite/granite-4.0-q4_k_m.gguf"
GRANITE_MODEL_F32_DEFAULT="${ROOT_DIR}/../agenticAiETH_layer4_Models/granite/granite-4.0.gguf"
if [[ -n "${GRANITE_MODEL_PATH:-}" ]]; then
  GRANITE_MODEL="${GRANITE_MODEL_PATH}"
elif [[ -f "${GRANITE_MODEL_QUANT_DEFAULT}" ]]; then
  # Prefer the quantized Granite build when available; it cuts latency significantly.
  GRANITE_MODEL="${GRANITE_MODEL_QUANT_DEFAULT}"
else
  GRANITE_MODEL="${GRANITE_MODEL_F32_DEFAULT}"
fi

PHI_PORT="${PHI_PORT:-8081}"
GRANITE_PORT="${GRANITE_PORT:-8082}"
LOCALAI_PORT="${LOCALAI_PORT:-8080}"

PHI_THREADS="${PHI_THREADS:-8}"
GRANITE_THREADS="${GRANITE_THREADS:-8}"
PHI_N_GPU_LAYERS="${PHI_N_GPU_LAYERS:-}"
GRANITE_N_GPU_LAYERS="${GRANITE_N_GPU_LAYERS:-24}"
GEMMA2B_MODEL_QUANT_DEFAULT="${ROOT_DIR}/../agenticAiETH_layer4_Models/gemma-2b-q4_k_m.gguf"
GEMMA2B_MODEL_PATH_DEFAULT="${ROOT_DIR}/../agenticAiETH_layer4_Models/gemma-2b.gguf"
if [[ -n "${GEMMA_MODEL_PATH:-}" ]]; then
  GEMMA2B_MODEL="${GEMMA_MODEL_PATH}"
elif [[ -f "${GEMMA2B_MODEL_QUANT_DEFAULT}" ]]; then
  # Prefer the quantized variant when available for faster inference.
  GEMMA2B_MODEL="${GEMMA2B_MODEL_QUANT_DEFAULT}"
else
  GEMMA2B_MODEL="${GEMMA2B_MODEL_PATH_DEFAULT}"
fi
GEMMA_PORT="${GEMMA_PORT:-8083}"
GEMMA_THREADS="${GEMMA_THREADS:-8}"
GEMMA_N_GPU_LAYERS="${GEMMA_N_GPU_LAYERS:--1}"
GEMMA7B_MODEL_PATH_DEFAULT="${ROOT_DIR}/../agenticAiETH_layer4_Models/gemma-7b.gguf"
GEMMA7B_MODEL="${GEMMA7B_MODEL_PATH:-${GEMMA7B_MODEL_PATH_DEFAULT}}"
GEMMA7B_PORT="${GEMMA7B_PORT:-8084}"
GEMMA7B_THREADS="${GEMMA7B_THREADS:-8}"
GEMMA7B_N_GPU_LAYERS="${GEMMA7B_N_GPU_LAYERS:--1}"
ENABLE_GEMMA7B="${ENABLE_GEMMA7B:-0}"
LOCALAI_CONFIG="${LOCALAI_CONFIG:-config/domains.json}"
DISABLE_VAULTGEMMA_FALLBACK="${DISABLE_VAULTGEMMA_FALLBACK:-1}"

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  echo "Stack appears to be running (found ${PID_FILE}). Stop it first with scripts/stop_localai_stack.sh." >&2
  exit 1
fi

LLAMA_BIN="${LLAMA_ROOT}/build/bin/llama-server"
if [[ ! -x "${LLAMA_BIN}" ]]; then
  echo "llama-server binary not found. Building with Metal backend..." >&2
  cmake -B "${LLAMA_ROOT}/build" -S "${LLAMA_ROOT}" -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON
  cmake --build "${LLAMA_ROOT}/build" --target llama-server --config Release
fi

if [[ ! -f "${PHI_MODEL}" ]]; then
  echo "Phi GGUF model missing at ${PHI_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${GRANITE_MODEL}" ]]; then
  echo "Granite GGUF model missing at ${GRANITE_MODEL}" >&2
  exit 1
fi

mkdir -p "${BIN_DIR}"
LOCALAI_BIN="${BIN_DIR}/vaultgemma-server"
if [[ ! -x "${LOCALAI_BIN}" ]]; then
  echo "Building vaultgemma-server binary..." >&2
  go build -o "${LOCALAI_BIN}" ./cmd/vaultgemma-server
fi

declare -a PIDS

start_llama() {
  local name=$1
  local port=$2
  local model=$3
  local threads=$4
  local n_gpu_layers=$5
  local log_file="${LOG_DIR}/${name}.log"

  echo "Starting llama.cpp server (${name}) on port ${port}..."
  cmd=( "${LLAMA_BIN}" --no-webui --ctx-size 4096 --threads "${threads}" --host 127.0.0.1 --port "${port}" --model "${model}" )
  if [[ -n "${n_gpu_layers}" ]]; then
    cmd+=( --n-gpu-layers "${n_gpu_layers}" )
  fi
  nohup "${cmd[@]}" > "${log_file}" 2>&1 &
  local pid=$!
  PIDS+=("${pid}:${log_file}")

  echo "  -> PID ${pid} (logs: ${log_file})"

  for _ in {1..180}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: llama.cpp server (${name}) exited early. Check ${log_file}" >&2
      exit 1
    fi
    if grep -q "main: model loaded" "${log_file}" 2>/dev/null; then
      return
    fi
    if grep -qi "error" "${log_file}" 2>/dev/null; then
      echo "ERROR: llama.cpp server (${name}) reported an error. Check ${log_file}" >&2
      exit 1
    fi
    sleep 1
  done

  echo "ERROR: llama.cpp server (${name}) did not finish loading within timeout. Check ${log_file}" >&2
  exit 1
}

start_llama "phi" "${PHI_PORT}" "${PHI_MODEL}" "${PHI_THREADS}" "${PHI_N_GPU_LAYERS}"
start_llama "granite" "${GRANITE_PORT}" "${GRANITE_MODEL}" "${GRANITE_THREADS}" "${GRANITE_N_GPU_LAYERS}"

if [[ -n "${GEMMA2B_MODEL}" && -f "${GEMMA2B_MODEL}" && -s "${GEMMA2B_MODEL}" ]]; then
  start_llama "gemma" "${GEMMA_PORT}" "${GEMMA2B_MODEL}" "${GEMMA_THREADS}" "${GEMMA_N_GPU_LAYERS}"
else
  echo "Gemma 2B model not found at ${GEMMA2B_MODEL}. Skipping Gemma 2B server."
fi

if [[ "${ENABLE_GEMMA7B}" == "1" ]]; then
  if [[ -f "${GEMMA7B_MODEL}" && -s "${GEMMA7B_MODEL}" ]]; then
    gemma7b_size=$(stat -f "%z" "${GEMMA7B_MODEL}" 2>/dev/null || echo 0)
    if [[ "${gemma7b_size}" -gt 1048576 ]]; then
      start_llama "gemma7b" "${GEMMA7B_PORT}" "${GEMMA7B_MODEL}" "${GEMMA7B_THREADS}" "${GEMMA7B_N_GPU_LAYERS}"
    else
      echo "Gemma 7B model at ${GEMMA7B_MODEL} appears truncated (${gemma7b_size} bytes). Skipping Gemma 7B server."
    fi
  else
    echo "Gemma 7B model not found at ${GEMMA7B_MODEL} or file empty. Skipping Gemma 7B server."
  fi
else
  echo "Gemma 7B server disabled. Set ENABLE_GEMMA7B=1 to launch it."
fi

echo "Starting LocalAI vaultgemma-server on port ${LOCALAI_PORT}..."
LOCALAI_LOG="${LOG_DIR}/localai.log"
nohup env DISABLE_VAULTGEMMA_FALLBACK="${DISABLE_VAULTGEMMA_FALLBACK}" ENABLE_GEMMA7B="${ENABLE_GEMMA7B}" "${LOCALAI_BIN}" -config "${LOCALAI_CONFIG}" -port "${LOCALAI_PORT}" \
  > "${LOCALAI_LOG}" 2>&1 &
LOCALAI_PID=$!
PIDS+=("${LOCALAI_PID}:${LOCALAI_LOG}")
echo "  -> PID ${LOCALAI_PID} (logs: ${LOCALAI_LOG})"

for _ in {1..180}; do
  if ! kill -0 "${LOCALAI_PID}" 2>/dev/null; then
    echo "ERROR: LocalAI server exited early. Check ${LOCALAI_LOG}" >&2
    exit 1
  fi
  if grep -q "Server ready on http://localhost:${LOCALAI_PORT}" "${LOCALAI_LOG}" 2>/dev/null; then
    break
  fi
  sleep 1
done

printf "%s\n" "${PIDS[@]}" > "${PID_FILE}"
echo "Stack started. PID file: ${PID_FILE}"
echo "To stop everything, run scripts/stop_localai_stack.sh"
