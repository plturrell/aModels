#!/usr/bin/env bash

# Quantize a transformer checkpoint into a GGUF file using llama.cpp helpers.
# Works entirely on CPU (no GPU required).
#
# Example:
#   ./scripts/quantize/quantize_model.sh \
#       --model ../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1 \
#       --quant q4_0
#
# Optional flags:
#   --llama ../third_party/llama.cpp   # path to llama.cpp checkout (overrides $LLAMA_CPP_DIR)
#   --output ./quantized               # directory for the resulting GGUF (defaults to model dir)
#   --tmp ./tmp                        # temp directory for intermediate fp16 GGUF (default: mktemp)
#   --quant q4_0                       # target quantization preset (q4_0, q4_1, q5_1, q8_0, …)

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: quantize_model.sh --model <model_path> [options]

Options:
  --model <path>     Path to original checkpoint directory (must contain config.json + safetensors).
  --quant <preset>   Quantization preset (default: q4_0).
  --llama <path>     Path to llama.cpp checkout (default: $LLAMA_CPP_DIR or ../third_party/llama.cpp).
  --output <dir>     Output directory for GGUF artifacts (default: model directory).
  --tmp <dir>        Temporary workspace (default: mktemp dir).
  -h|--help          Show this help.
EOF
}

MODEL=""
LLAMA_DIR="${LLAMA_CPP_DIR:-}"
QUANT="q4_0"
OUTPUT_DIR=""
TMPDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --quant)
      QUANT="$2"
      shift 2
      ;;
    --llama)
      LLAMA_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --tmp)
      TMPDIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Missing --model" >&2
  usage
  exit 1
fi

if [[ ! -d "$MODEL" ]]; then
  echo "Model path does not exist: $MODEL" >&2
  exit 1
fi

if [[ -z "$LLAMA_DIR" ]]; then
  LLAMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../third_party/llama.cpp" 2>/dev/null && pwd || true)"
fi

if [[ -z "$LLAMA_DIR" || ! -d "$LLAMA_DIR" ]]; then
  echo "Unable to locate llama.cpp checkout. Set LLAMA_CPP_DIR or pass --llama." >&2
  exit 1
fi

if [[ -f "$LLAMA_DIR/convert.py" ]]; then
  CONVERT_PY="$LLAMA_DIR/convert.py"
elif [[ -f "$LLAMA_DIR/convert_hf_to_gguf.py" ]]; then
  CONVERT_PY="$LLAMA_DIR/convert_hf_to_gguf.py"
else
  CONVERT_PY=""
fi
QUANT_BIN=""
if [[ -x "$LLAMA_DIR/llama-quantize" ]]; then
  QUANT_BIN="$LLAMA_DIR/llama-quantize"
elif [[ -x "$LLAMA_DIR/bin/llama-quantize" ]]; then
  QUANT_BIN="$LLAMA_DIR/bin/llama-quantize"
elif [[ -x "$LLAMA_DIR/build/bin/llama-quantize" ]]; then
  QUANT_BIN="$LLAMA_DIR/build/bin/llama-quantize"
elif [[ -x "$LLAMA_DIR/quantize" ]]; then
  QUANT_BIN="$LLAMA_DIR/quantize"
fi

if [[ -z "$CONVERT_PY" ]]; then
  echo "No convert script found in $LLAMA_DIR (expected convert.py or convert_hf_to_gguf.py)." >&2
  exit 1
fi

if [[ -z "$QUANT_BIN" ]]; then
  echo "llama-quantize binary not found. Build it with 'cmake -S . -B build && cmake --build build --target llama-quantize'." >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$MODEL"
fi
mkdir -p "$OUTPUT_DIR"

if [[ -z "$TMPDIR" ]]; then
  TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t quantize)"
  trap 'rm -rf "$TMPDIR"' EXIT
else
  mkdir -p "$TMPDIR"
fi

MODEL_NAME="$(basename "$MODEL")"
F16_GGUF="$TMPDIR/${MODEL_NAME}-fp16.gguf"
OUT_GGUF="$OUTPUT_DIR/${MODEL_NAME}-${QUANT}.gguf"

echo "==> Converting $MODEL to fp16 GGUF..."
if [[ "$(basename "$CONVERT_PY")" == "convert.py" ]]; then
  python3 "$CONVERT_PY" \
    --model-path "$MODEL" \
    --outfile "$F16_GGUF" \
    --format gguf
else
  python3 "$CONVERT_PY" \
    "$MODEL" \
    --outfile "$F16_GGUF" \
    --outtype f16
fi

echo "==> Quantizing to $QUANT -> $OUT_GGUF"
"$QUANT_BIN" "$F16_GGUF" "$OUT_GGUF" "$QUANT"

echo "==> Done."
echo "Quantized model written to: $OUT_GGUF"
