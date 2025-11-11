# Transformers CPU Service

Some domains now use HuggingFace transformers directly (Phi-3.5-mini and Granite 4.0).
A lightweight FastAPI server at `services/transformers_cpu_server.py` exposes a
`/v1/generate` endpoint that the Go runtime calls whenever a domain has
`"backend_type": "hf-transformers"` in `config/domains.json`.

## Requirements

```bash
python3 -m pip install --user fastapi uvicorn transformers torch sentencepiece safetensors
```

The service loads models from the local checkout under
`agenticAiETH_layer4_Models/phi/...` and `agenticAiETH_layer4_Models/granite/...`.
Adjust `MODEL_REGISTRY` in the script if paths change.

## Running

```bash
cd agenticAiETH_layer4_LocalAI/services
uvicorn transformers_cpu_server:app --host 0.0.0.0 --port 9090
```

Environment variable `TRANSFORMERS_CPU_PORT` can override the default port.

## API Contract

POST `/v1/generate`

```json
{
  "model": "phi-3.5-mini",
  "prompt": "Summarize...",
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 0
}
```

Response:

```json
{"text": "...", "tokens": 42}
```

## Integrating New Domains

1. Add the model checkpoint under `agenticAiETH_layer4_Models/`.
2. Extend `MODEL_REGISTRY` in the Python service.
3. Update `config/domains.json` with

```json
"backend_type": "hf-transformers",
"model_name": "your-model-key",
"model_path": "",
"transformers_config": {
  "endpoint": "http://localhost:9090/v1/generate",
  "model_name": "your-model-key",
  "timeout_seconds": 180
}
```
4. Restart the Python service and the Go server.
