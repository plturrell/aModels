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


def _allocate_gpu_for_ocr():
    """Allocate GPU from orchestrator if available."""
    gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATOR_URL")
    if not gpu_orchestrator_url:
        return "cuda" if torch.cuda.is_available() else "cpu", None
    
    try:
        import httpx
        request_data = {
            "service_name": "deepseek-ocr-extract",
            "workload_type": "ocr",
            "workload_data": {"image_count": 1}
        }
        
        response = httpx.post(
            f"{gpu_orchestrator_url}/gpu/allocate",
            json=request_data,
            timeout=10.0
        )
        
        if response.status_code == 200:
            allocation = response.json()
            device_ids = allocation.get("gpu_ids", [0])
            # Set CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
            return "cuda", allocation.get("id")
        else:
            return "cuda" if torch.cuda.is_available() else "cpu", None
    except Exception as e:
        print(f"Warning: Failed to allocate GPU from orchestrator: {e}", file=sys.stderr)
        return "cuda" if torch.cuda.is_available() else "cpu", None


def run_inference(image_path: Path, prompt: str, fmt: str) -> str:
    device, allocation_id = _allocate_gpu_for_ocr()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )

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
    
    # Store allocation_id for cleanup
    run_inference._allocation_id = allocation_id

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
    
    # Release GPU allocation if allocated
    allocation_id = getattr(run_inference, '_allocation_id', None)
    if allocation_id:
        gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATOR_URL")
        if gpu_orchestrator_url:
            try:
                import httpx
                httpx.post(
                    f"{gpu_orchestrator_url}/gpu/release",
                    json={"allocation_id": allocation_id},
                    timeout=5.0
                )
            except Exception:
                pass  # Ignore cleanup errors
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
