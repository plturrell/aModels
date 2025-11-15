# NLLB-200 Machine Translation Setup

This document describes the setup for NLLB-200 (No Language Left Behind) machine translation models for English ↔ Arabic translation on your T4 GPU.

## Overview

The setup includes:
- **Primary MT Model**: NLLB-200 1.3B (recommended for T4 GPU, best quality)
- **Fallback Model**: NLLB-200 Distilled 600M (lighter, good quality)
- **Optional Backup**: M2M100-418M (lighter, high-throughput scenarios)

## Architecture

### Components

1. **Transformers Service** (`services/transformers_cpu_server.py`)
   - Extended to support seq2seq (translation) models
   - New `/v1/translate` endpoint for translation requests
   - Handles both NLLB-200 and M2M100 models

2. **Domain Configuration** (`config/domains.json`)
   - `mt-en-ar`: English → Arabic translation
   - `mt-ar-en`: Arabic → English translation
   - `mt-en-ar-light`: Lightweight English → Arabic (M2M100)

3. **Routing Rules** (`config/routing_rules.json`)
   - Routes `/v1/translate/en-ar` → `mt-en-ar` domain
   - Routes `/v1/translate/ar-en` → `mt-ar-en` domain
   - Keyword-based routing for translation queries

## Setup Steps

### 1. Download Models

Run the download script:

```bash
cd /home/aModels/services/localai/scripts
./download_nllb_models.sh
```

This will download:
- `facebook/nllb-200-distilled-600M` → `/home/aModels/infrastructure/models/nllb-200-distilled-600M`
- `facebook/nllb-200-1.3B` → `/home/aModels/infrastructure/models/nllb-200-1.3B`
- Optionally: M2M100 models

### 2. Update Docker Configuration

Ensure your transformers service container:
- Has GPU access (`--gpus all` or `runtime: nvidia`)
- Mounts `/home/aModels/infrastructure/models` to `/models` in the container
- Has the updated `requirements-transformers.txt` (includes `sentencepiece`)

Example docker-compose snippet:

```yaml
transformers-service:
  build:
    context: .
    dockerfile: Dockerfile.transformers
  volumes:
    - /home/aModels/infrastructure/models:/models
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### 3. Restart Services

```bash
# Restart transformers service
docker-compose restart transformers-service

# Or rebuild if needed
docker-compose build transformers-service
docker-compose up -d transformers-service
```

### 4. Verify Setup

Check that models are accessible:

```bash
docker exec transformers-service ls -lh /models/nllb-200-1.3B
```

Test the translation endpoint:

```bash
curl -X POST http://localhost:9090/v1/translate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nllb-200-1.3B",
    "text": "Hello world",
    "source_lang": "eng_Latn",
    "target_lang": "arb_Arab"
  }'
```

## Usage

### Direct API Calls

#### English to Arabic

```bash
curl -X POST http://transformers-service:9090/v1/translate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nllb-200-1.3B",
    "text": "The financial report shows a profit of $1,000,000.",
    "source_lang": "eng_Latn",
    "target_lang": "arb_Arab",
    "max_length": 512
  }'
```

#### Arabic to English

```bash
curl -X POST http://transformers-service:9090/v1/translate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nllb-200-1.3B",
    "text": "التقرير المالي يظهر ربحاً قدره مليون دولار.",
    "source_lang": "arb_Arab",
    "target_lang": "eng_Latn",
    "max_length": 512
  }'
```

### Via LocalAI Routing

Once routing is configured, you can use:

```bash
# English to Arabic
curl -X POST http://localai:8080/v1/translate/en-ar \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The financial report shows a profit of $1,000,000."
  }'

# Arabic to English
curl -X POST http://localai:8080/v1/translate/ar-en \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "التقرير المالي يظهر ربحاً قدره مليون دولار."
  }'
```

## Language Codes

### NLLB-200 Language Codes
- English: `eng_Latn`
- Arabic (Modern Standard): `arb_Arab`

### M2M100 Language Codes
- English: `en`
- Arabic: `ar`

## Model Specifications

### NLLB-200 1.3B
- **Size**: ~2.6 GB (model weights)
- **VRAM**: ~4-6 GB (with float16)
- **Quality**: Excellent for formal/regulatory text
- **Speed**: ~10-20 tokens/sec on T4

### NLLB-200 Distilled 600M
- **Size**: ~1.2 GB
- **VRAM**: ~2-3 GB (with float16)
- **Quality**: Good, slightly lower than 1.3B
- **Speed**: ~20-30 tokens/sec on T4

### M2M100-418M
- **Size**: ~800 MB
- **VRAM**: ~1.5-2 GB
- **Quality**: Adequate for high-throughput scenarios
- **Speed**: ~30-40 tokens/sec on T4

## Best Practices

### For Financial/Regulatory Text

1. **Use NLLB-200 1.3B** for best quality
2. **Set low temperature** (0.1) to ensure consistency
3. **Use beam search** (num_beams=5) for better quality
4. **Post-process validation**: Run translated text through your existing LLM (Phi/Gemma) to:
   - Check glossary compliance
   - Verify numbers/dates are preserved
   - Flag suspicious changes

### Prompt Discipline

For every translation request, wrap with strict instructions:

```
Translate from English to Modern Standard Arabic. 
Preserve numbers, dates, amounts, and references exactly. 
Do not explain. Output only the translated text.
```

## Troubleshooting

### Model Not Found

If you see "model path does not exist":
1. Verify models are downloaded: `ls -lh /home/aModels/infrastructure/models/nllb-200-1.3B`
2. Check Docker volume mount: `docker exec transformers-service ls -lh /models`
3. Verify MODEL_REGISTRY in `transformers_cpu_server.py`

### Out of Memory

If you get CUDA OOM errors:
1. Use `nllb-200-distilled-600M` instead of `1.3B`
2. Reduce batch size (currently processes one text at a time)
3. Use `float16` (already enabled for GPU)

### Poor Translation Quality

1. Ensure correct language codes are used
2. For financial terms, consider creating a custom glossary
3. Use the 1.3B model instead of distilled version
4. Increase `max_length` for longer sentences

## Integration with Existing Services

The translation domains are integrated into your existing LocalAI infrastructure:

- **Domain Manager**: Automatically loads MT domains from `domains.json`
- **Routing**: Translation requests are routed via `routing_rules.json`
- **Fallback**: Falls back to lighter models if primary model is unavailable
- **Monitoring**: Translation requests are logged like other domain requests

## Next Steps

1. **Download models** using the provided script
2. **Test translation** with sample financial/regulatory text
3. **Integrate** into your application workflows
4. **Monitor** translation quality and adjust as needed
5. **Consider** fine-tuning on domain-specific data if needed

## References

- [NLLB-200 Model Card](https://huggingface.co/facebook/nllb-200-1.3B)
- [NLLB-200 Paper](https://arxiv.org/abs/2207.04672)
- [M2M100 Model Card](https://huggingface.co/facebook/m2m100_418M)

