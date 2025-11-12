# DeepSeek OCR Benchmark

This benchmark evaluates the `DeepSeek-OCR` vision model by submitting document images to the OCR endpoint and comparing the extracted text with ground-truth annotations.

## Dataset Format

The benchmark expects a JSON Lines (`.jsonl`) file where each record provides the path to an image and the expected transcription:

```json
{"image": "docs/sample_invoice.png", "text": "Acme Inc.\nInvoice #12345\n...", "prompt": "<image> Extract the textual content as Markdown."}
```

- `image` – path to the image file (absolute or relative to the dataset directory)
- `text` – ground-truth transcription
- `prompt` *(optional)* – custom prompt to send to the OCR model

When the `--data` flag points to a directory, the runner looks for `annotations.jsonl` inside that directory. Otherwise the flag must reference the JSONL file directly.

## Running the Benchmark

```bash
DEEPSEEK_OCR_ENDPOINT=http://localhost:9393/v1/ocr \
DEEPSEEK_OCR_MODEL=deepseek-ai/DeepSeek-OCR \
go run ./cmd/aibench run \
  --task=deepseek_ocr \
  --data=/path/to/ocr_dataset \
  --limit=50
```

### Environment Variables

- `DEEPSEEK_OCR_ENDPOINT` – OCR HTTP endpoint (default `http://localhost:9393/v1/ocr`)
- `DEEPSEEK_OCR_API_KEY` – optional bearer token for the endpoint
- `DEEPSEEK_OCR_MODEL` – optional model variant to request (sent as `model` field)
- `DEEPSEEK_OCR_PROMPT` – default prompt when the dataset item does not specify one
- `DEEPSEEK_OCR_TIMEOUT` – request timeout (e.g. `90s` or `120000` for milliseconds)

## Metrics

The runner reports:

- `cer` – character error rate (aggregated across samples)
- `wer` – word error rate
- `exact_match` – fraction of samples with exact (normalized) match
- `char_total`, `word_total` – denominators used for CER/WER

Detailed per-sample results are included when `--limit` is small or when the run writes to `--out`.
