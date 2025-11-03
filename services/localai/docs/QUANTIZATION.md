# Quantizing LocalAI Models

This project now includes a repeatable CPU-only pipeline for turning full-precision checkpoints
into GGUF quantized weights. Use it any time a new model needs to run faster or fit into less
memory on the local server.

## Prerequisites

1. **llama.cpp checkout** – clone https://github.com/ggerganov/llama.cpp somewhere inside the repo
   (recommended: `third_party/llama.cpp`). Build the `quantize` binary once:
   ```bash
   cmake -S third_party/llama.cpp -B third_party/llama.cpp/build
   cmake --build third_party/llama.cpp/build --target quantize
   ```
   The helper script relies on `convert.py` and the compiled `quantize` binary.
2. **Python 3** – for running `convert.py`.

## Quantizing a Model

From `agenticAiETH_layer4_LocalAI` run:
```bash
./scripts/quantize/quantize_model.sh \
  --model ../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1 \
  --quant q4_0
```

The script will:
1. Convert the original safetensors checkpoint to an intermediate fp16 GGUF file.
2. Quantize it with llama.cpp to the requested preset (default `q4_0`).
3. Drop the quantized GGUF next to the original model directory: e.g.
   `vaultgemma-transformers-1b-v1-q4_0.gguf`.

Key flags:
- `--quant q5_1` – choose a different preset (q4_1, q5_1, q8_0, etc.).
- `--llama /path/to/llama.cpp` – override the auto-detected llama.cpp location.
- `--output ./quantized` – write GGUF files to a dedicated folder.

## Updating LocalAI Domains

After producing the GGUF, point each domain entry in
`agenticAiETH_layer4_LocalAI/config/domains.json` to the quantized file, for example:
```json
"model_path": "../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1-q4_0.gguf"
```

Restart `vaultgemma-server` and it will load the smaller, faster model. Repeat the process for
Phi-3.5, Granite, or any new checkpoints you add under `agenticAiETH_layer4_Models/`.

## Batch Quantization

To quantize everything under `agenticAiETH_layer4_Models`:
```bash
for model_dir in ../agenticAiETH_layer4_Models/*; do
  if [[ -d "$model_dir" ]]; then
    ./scripts/quantize/quantize_model.sh --model "$model_dir" --quant q4_0
  fi
done
```

Adjust the loop or preset as needed (e.g. run twice for `q4_0` and `q5_1`).  The process is
entirely CPU bound – no GPU access is required.
