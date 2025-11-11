#!/usr/bin/env bash
#
# Helper to launch the official llama.cpp HTTP server that was built in
# third_party/llama.cpp.  Usage:
#   scripts/run_llama_cpp_server.sh <path-to-model.gguf> [-- additional args]
#
# Examples:
#   scripts/run_llama_cpp_server.sh \
#     ../agenticAiETH_layer4_Models/phi/phi-3-pytorch-phi-3.5-mini-instruct-v2-q4_0.gguf \
#     --port 8081 --ctx-size 4096
#
#   scripts/run_llama_cpp_server.sh \
#     ../agenticAiETH_layer4_Models/granite/granite-4.0-transformers-granite-q4_0.gguf
#
# This script validates that the binary exists and reuses sensible defaults for
# host/port while allowing additional llama-server arguments to be appended.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${ROOT_DIR}/../third_party/llama.cpp"
LLAMA_BIN="${LLAMA_ROOT}/bin/llama-server"
if [[ ! -x "${LLAMA_BIN}" ]]; then
  LLAMA_BIN="${LLAMA_ROOT}/build/bin/llama-server"
fi

if [[ ! -x "${LLAMA_BIN}" ]]; then
  echo "error: llama-server binary not found. Build it with:" >&2
  echo "    cd third_party/llama.cpp" >&2
  echo "    cmake -B build -S . -DLLAMA_BUILD_SERVER=ON -DGGML_METAL=ON" >&2
  echo "    cmake --build build --target llama-server --config Release" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage: run_llama_cpp_server.sh <path-to-model.gguf> [llama-server args...]

Examples:
  run_llama_cpp_server.sh ../agenticAiETH_layer4_Models/phi/phi-3-pytorch-phi-3.5-mini-instruct-v2-q4_0.gguf
  run_llama_cpp_server.sh ../agenticAiETH_layer4_Models/granite/granite-4.0-transformers-granite-q4_0.gguf --port 8082 --ctx-size 4096
USAGE
  exit 1
fi

MODEL_PATH="$1"
shift

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "error: model file '${MODEL_PATH}' does not exist." >&2
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
THREADS="${THREADS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"

echo "Launching llama-server with model:"
echo "  ${MODEL_PATH}"
echo "Host: ${HOST}  Port: ${PORT}  Threads: ${THREADS}"
echo "Additional arguments: $*"
echo

exec "${LLAMA_BIN}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --threads "${THREADS}" \
  --model "${MODEL_PATH}" \
  "$@"
