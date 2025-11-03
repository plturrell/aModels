#!/usr/bin/env python3
"""
CLI wrapper around DeepSeek-OCR huggingface deployment.

This script expects `transformers`, `torch`, and the DeepSeek-OCR model weights
to be available locally (they will be downloaded on first run).

Usage:
  python deepseek_ocr_cli.py --input /path/to/image.png --output /path/to/output.md
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - runtime dependency
    print(f"DeepSeek OCR dependencies missing: {exc}. Install transformers & torch.", file=sys.stderr)
    sys.exit(2)


MODEL_NAME = os.environ.get("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_inference(image_path: Path, prompt: str, fmt: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)

    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(image_path),
        output_path=None,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=False,
    )

    if isinstance(result, dict) and "text" in result:
        text = result["text"]
    elif isinstance(result, (list, tuple)):
        text = "\n\n".join(str(item) for item in result)
    else:
        text = str(result)

    if fmt == "json":
        return json.dumps({"text": text}, ensure_ascii=False, indent=2)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR CLI wrapper")
    parser.add_argument("--input", required=True, help="Path to image or PDF file")
    parser.add_argument("--output", required=True, help="Path to output markdown/json file")
    parser.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown.", help="Prompt fed to the model")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"Input file {input_path} not found", file=sys.stderr)
        return 1

    ensure_parent(output_path)

    text = run_inference(input_path, args.prompt, args.format)

    if args.format == "markdown" and not output_path.suffix:
        output_path = output_path.with_suffix(".md")
    elif args.format == "json" and output_path.suffix != ".json":
        output_path = output_path.with_suffix(".json")

    output_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
