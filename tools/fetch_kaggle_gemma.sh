#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "KAGGLE_USERNAME and KAGGLE_KEY must be set in your environment." >&2
  echo "Generate a token from https://www.kaggle.com/settings/account and export the values," >&2
  echo "e.g. export KAGGLE_USERNAME=your_user && export KAGGLE_KEY=your_key" >&2
  exit 1
fi

MODEL_DIR="${MODEL_DIR:-$PWD/models}"
mkdir -p "$MODEL_DIR"

output="${MODEL_DIR}/gemma-2b-gguf.tar.gz"

curl -L -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
  -o "$output" \
  "https://www.kaggle.com/api/v1/models/google/gemma/gguf/2b/1/download"

echo "Downloaded Gemma 2B GGUF to $output"

echo "To extract:" >&2
echo "  tar -xvzf $output -C $MODEL_DIR" >&2
