"""Simple FastAPI service for running HuggingFace transformers on CPU.

Launch with:

    uvicorn transformers_cpu_server:app --host 0.0.0.0 --port 9090

The service loads the Phi-3.5-mini and Granite 4.0 checkpoints from the
local repository and exposes a `/v1/generate` endpoint that accepts
JSON requests of the form:

    {
        "model": "phi-3.5-mini",
        "prompt": "...",
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0
    }

The response contains the generated text and token count:

    {"text": "...", "tokens": 42}
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "phi-3.5-mini": {
        "path": os.path.join(
            BASE_DIR,
            "agenticAiETH_layer4_Models",
            "phi",
            "phi-3-pytorch-phi-3.5-mini-instruct-v2",
        )
    },
    "granite-4.0-h-micro": {
        "path": os.path.join(
            BASE_DIR,
            "agenticAiETH_layer4_Models",
            "granite",
            "granite-4.0-transformers-granite",
        )
    },
}

app = FastAPI(title="Transformers CPU Service", version="0.1.0")
torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 1))


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0


class GenerateResponse(BaseModel):
    text: str
    tokens: int


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0


def _format_chat_prompt(messages: List[ChatMessage]) -> str:
    """
    Convert OpenAI-style chat messages into a single prompt string.
    """
    if not messages:
        return ""

    system_prefix = ""
    parts: List[str] = []

    for message in messages:
        role = message.role.strip().lower()
        content = message.content.strip()
        if not content:
            continue

        if role == "system":
            system_prefix = content
            continue

        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"{role.capitalize()}: {content}")

    prompt_body = "\n".join(parts)
    if system_prefix:
        return f"{system_prefix}\n\n{prompt_body}\nAssistant:"
    return f"{prompt_body}\nAssistant:"


def _run_generation(req: GenerateRequest) -> GenerateResponse:
    print(f"[DEBUG] Loading tokenizer for {req.model}")
    tokenizer = load_tokenizer(req.model)
    print(f"[DEBUG] Loading model for {req.model}")
    model = load_model(req.model)

    print(f"[DEBUG] Tokenizing prompt: {req.prompt[:50]}...")
    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"[DEBUG] Input shape: {input_ids.shape}")

    max_new_tokens = max(1, min(req.max_tokens, 512))
    temperature = req.temperature if req.temperature > 0 else 0.7
    top_p = req.top_p if 0 < req.top_p <= 1 else 0.9
    top_k = max(0, req.top_k)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,  # Use KV cache for faster generation
        "num_beams": 1,  # Greedy decoding for speed
    }
    if top_k > 0:
        generation_kwargs["top_k"] = top_k

    print(f"[DEBUG] Starting generation with kwargs: {generation_kwargs}")
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    print(f"[DEBUG] Generation complete, decoding...")
    generated_ids = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    token_count = len(tokenizer.tokenize(text))

    print(f"[DEBUG] Generated {token_count} tokens")
    return GenerateResponse(text=text.strip(), tokens=token_count)


@lru_cache(maxsize=None)
def load_tokenizer(model_key: str):
    cfg = MODEL_REGISTRY.get(model_key)
    if not cfg:
        raise ValueError(f"unknown model: {model_key}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    return tokenizer


@lru_cache(maxsize=None)
def load_model(model_key: str):
    cfg = MODEL_REGISTRY.get(model_key)
    if not cfg:
        raise ValueError(f"unknown model: {model_key}")
    
    print(f"[DEBUG] Loading model from {cfg['path']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    print("[DEBUG] Model loaded successfully")
    return model


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        return _run_generation(req)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    prompt = _format_chat_prompt(req.messages)
    gen_req = GenerateRequest(
        model=req.model,
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    try:
        tokenizer = load_tokenizer(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    prompt_token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_tokens = prompt_token_ids.shape[1]

    try:
        generation = _run_generation(gen_req)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    created = int(time.time())
    total_tokens = prompt_tokens + generation.tokens

    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generation.text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": generation.tokens,
            "total_tokens": int(total_tokens),
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "models": list(MODEL_REGISTRY.keys())}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TRANSFORMERS_CPU_PORT", "9090"))
    uvicorn.run("transformers_cpu_server:app", host="0.0.0.0", port=port, reload=False)
