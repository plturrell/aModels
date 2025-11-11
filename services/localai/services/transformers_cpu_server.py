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
import threading
import time
from functools import lru_cache
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import quantization libraries
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    BitsAndBytesConfig = None

# Use /models mount point from Docker volume, or fallback to absolute path
MODELS_BASE = os.getenv("MODELS_BASE", "/models")
# If MODELS_BASE doesn't exist, try absolute path from aModels root
if not os.path.exists(MODELS_BASE):
    # Try absolute path: /home/aModels/models
    absolute_models = "/home/aModels/models"
    if os.path.exists(absolute_models):
        MODELS_BASE = absolute_models
        print(f"✅ Using absolute models path: {MODELS_BASE}")
    else:
        # Try relative path: ../../models from services/localai/services/
        relative_models = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "..", "models")
        if os.path.exists(relative_models):
            MODELS_BASE = os.path.abspath(relative_models)
            print(f"✅ Using relative models path: {MODELS_BASE}")
else:
    print(f"✅ Using MODELS_BASE from environment: {MODELS_BASE}")

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "phi-3.5-mini": {
        "path": os.path.join(
            MODELS_BASE,
            "phi-3.5-mini-instruct-pytorch",
        )
    },
    "granite-4.0-h-micro": {
        "path": os.path.join(
            MODELS_BASE,
            "granite",
            "granite-4.0-transformers-granite",
        )
    },
    "vaultgemma": {
        "path": os.path.join(
            MODELS_BASE,
            "vaultgemma-1b-transformers",
        )
    },
}

app = FastAPI(title="Transformers GPU Service", version="0.1.0")
torch.set_grad_enabled(False)

# Detect device (GPU if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0

# Quantization setting from environment
QUANTIZATION = os.getenv("TRANSFORMERS_QUANTIZATION", "none").lower()

# Model access tracking for eviction
_model_access_times: Dict[str, datetime] = {}
_model_access_lock = threading.Lock()
MODEL_UNLOAD_TIMEOUT = int(os.getenv("MODEL_UNLOAD_TIMEOUT_SECONDS", "3600"))  # 1 hour default

if DEVICE == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Number of GPUs: {NUM_GPUS}")
    if QUANTIZATION != "none":
        print(f"   Quantization: {QUANTIZATION}")
else:
    print("⚠️  CUDA not available, using CPU")
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


def _update_model_access(model_key: str):
    """Update last access time for a model"""
    with _model_access_lock:
        _model_access_times[model_key] = datetime.now()

def _unload_unused_models():
    """Unload models that haven't been accessed recently"""
    cutoff = datetime.now() - timedelta(seconds=MODEL_UNLOAD_TIMEOUT)
    with _model_access_lock:
        to_unload = [
            key for key, access_time in _model_access_times.items()
            if access_time < cutoff
        ]
    
    for model_key in to_unload:
        # Note: lru_cache doesn't support manual eviction easily
        # In production, you might want to use a custom cache implementation
        print(f"[DEBUG] Model {model_key} would be unloaded (not accessed for {MODEL_UNLOAD_TIMEOUT}s)")

def _run_generation(req: GenerateRequest) -> GenerateResponse:
    print(f"[DEBUG] Loading tokenizer for {req.model}")
    tokenizer = load_tokenizer(req.model)
    print(f"[DEBUG] Loading model for {req.model}")
    model = load_model(req.model)
    _update_model_access(req.model)

    print(f"[DEBUG] Tokenizing prompt: {req.prompt[:50]}...")
    inputs = tokenizer(req.prompt, return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    print(f"[DEBUG] Input shape: {input_ids.shape}, Device: {DEVICE}")

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


def _get_device_map() -> str:
    """Get device map for multi-GPU support"""
    if DEVICE != "cuda" or NUM_GPUS == 0:
        return "cpu"
    if NUM_GPUS == 1:
        return "cuda:0"
    # For multi-GPU, use auto device_map to distribute layers
    return "auto"

def _get_quantization_config():
    """Get quantization configuration if enabled"""
    if not HAS_BITSANDBYTES or QUANTIZATION == "none":
        return None
    
    if QUANTIZATION == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif QUANTIZATION == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True
        )
    return None

@lru_cache(maxsize=None)
def load_model(model_key: str):
    cfg = MODEL_REGISTRY.get(model_key)
    if not cfg:
        raise ValueError(f"unknown model: {model_key}")
    
    print(f"[DEBUG] Loading model from {cfg['path']}")
    
    # Use appropriate dtype and device for GPU
    if DEVICE == "cuda":
        # Use float16 for GPU to save memory and speed up inference
        torch_dtype = torch.float16
        device_map = _get_device_map()
        print(f"[DEBUG] Loading on {device_map} with float16")
    else:
        torch_dtype = torch.float32
        device_map = "cpu"
        print(f"[DEBUG] Loading on CPU with float32")
    
    quantization_config = _get_quantization_config()
    if quantization_config:
        print(f"[DEBUG] Using quantization: {QUANTIZATION}")
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.eval()
    
    # Update access time
    with _model_access_lock:
        _model_access_times[model_key] = datetime.now()
    
    print(f"[DEBUG] Model loaded successfully on {device_map}")
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
    if DEVICE == "cuda":
        prompt_token_ids = prompt_token_ids.to(DEVICE)
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
    gpu_info = {}
    if DEVICE == "cuda":
        gpu_info = {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
        }
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu": gpu_info,
        "models": list(MODEL_REGISTRY.keys())
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TRANSFORMERS_CPU_PORT", "9090"))
    uvicorn.run("transformers_cpu_server:app", host="0.0.0.0", port=port, reload=False)
