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
import logging

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# Import metrics
try:
    from toon_metrics import get_metrics_collector
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger = logging.getLogger(__name__)
    logger.warning("Metrics module not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import model client for network-based model access
try:
    from model_client import get_model_path
    MODEL_SERVER_ENABLED = True
except ImportError:
    MODEL_SERVER_ENABLED = False
    def get_model_path(model_name: str, registry_path: str) -> str:
        return registry_path

# Model registry - paths are model directory names on model-server
# All models are fetched from model-server and cached locally
# This registry maps model names used in domains.json to actual model directory names
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # Core models used by LocalAI domains
    "phi-3.5-mini": {
        "path": "phi-3.5-mini-instruct-pytorch"
    },
    "granite-4.0-h-micro": {
        "path": "granite-4.0-h-micro-transformers"
    },
    "granite-4.0": {
        "path": "granite-4.0-h-micro-transformers"  # Alias to h-micro
    },
    "vaultgemma-1b": {
        "path": "vaultgemma-1b-transformers"
    },
    "calm-l": {
        "path": "calm/hf/CALM-L"
    },
    "deepseek-ocr": {
        "path": "DeepSeek-OCR"
    },
    "sap-rpt-1": {
        "path": "sap-rpt-1-oss-main"
    },
    "cwm": {
        "path": "cwm"
    },
    # Translation models (NLLB-200)
    "nllb-200-distilled-600M": {
        "path": "arabic_models/nllb-200-distilled-600M",
        "type": "seq2seq",
        "src_lang": "eng_Latn",
        "tgt_lang": "arb_Arab"
    },
    "nllb-200-1.3B": {
        "path": "arabic_models/nllb-200-1.3B",
        "type": "seq2seq",
        "src_lang": "eng_Latn",
        "tgt_lang": "arb_Arab"
    },
    # M2M100 models (optional lighter backup)
    "m2m100-418M": {
        "path": "arabic_models/m2m100-418M",
        "type": "seq2seq",
        "src_lang": "en",
        "tgt_lang": "ar"
    },
    "m2m100-1.2B": {
        "path": "arabic_models/m2m100-1.2B",
        "type": "seq2seq",
        "src_lang": "en",
        "tgt_lang": "ar"
    },
    # Arabic NLP models for TOON generation
    "kuwain-1.5B": {
        "path": "arabic_models/kuwain-1.5B",
        "type": "causal",
        "description": "Arabic language model for TOON generation"
    },
    "qadi": {
        "path": "arabic_models/qadi",
        "type": "classifier",
        "description": "Dialect classification model"
    },
    # ARBML tools are Python packages, not model files
    # They will be used via camel-tools library
    # Note: gemma-2b-it and gemma-7b-it use TensorRT backends, not hf-transformers
    # They are handled by LocalAI's TensorRT backend directly
}

app = FastAPI(title="Transformers GPU Service", version="0.1.0")

# Initialize metrics collector
if METRICS_ENABLED:
    metrics = get_metrics_collector()
    logger.info("Metrics collection enabled")
else:
    metrics = None
    logger.info("Metrics collection disabled")
torch.set_grad_enabled(False)

# Detect device (GPU if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
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


class TranslateRequest(BaseModel):
    model: str
    text: str
    source_lang: str | None = None  # Optional, will use model default if not provided
    target_lang: str | None = None  # Optional, will use model default if not provided
    max_length: int = 512
    use_toon: bool = True  # Enable TOON-enhanced translation by default (falls back to direct if TOON fails)
    toon_config: Dict[str, Any] | None = None  # TOON-specific configuration


class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    tokens: int
    toon_used: bool = False  # Whether TOON was used
    toon_data: Dict[str, Any] | None = None  # TOON representation if used


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
    
    # Try to get model path from model-server or direct volume
    try:
        model_path = get_model_path(model_key, cfg["path"])
    except (ValueError, Exception) as e:
        print(f"[DEBUG] get_model_path failed: {e}, trying direct volume access...")
        model_path = None
    
    # If get_model_path failed or path doesn't exist, try direct volume access
    if not model_path or not os.path.exists(model_path):
        # Try direct paths
        alt_paths = [
            os.path.join("/models", cfg["path"]),
            cfg["path"],
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"[DEBUG] Found model at direct path: {model_path}")
                break
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model path does not exist. Tried: {[cfg['path']] + alt_paths}")
    
    print(f"[DEBUG] Loading tokenizer for {model_key} from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


@lru_cache(maxsize=None)
def load_model(model_key: str):
    cfg = MODEL_REGISTRY.get(model_key)
    if not cfg:
        raise ValueError(f"unknown model: {model_key}")
    
    # Try to get model path from model-server or direct volume
    try:
        model_path = get_model_path(model_key, cfg["path"])
    except (ValueError, Exception) as e:
        print(f"[DEBUG] get_model_path failed: {e}, trying direct volume access...")
        model_path = None
    
    # If get_model_path failed or path doesn't exist, try direct volume access
    if not model_path or not os.path.exists(model_path):
        # Try direct paths
        alt_paths = [
            os.path.join("/models", cfg["path"]),
            cfg["path"],
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"[DEBUG] Found model at direct path: {model_path}")
                break
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model path does not exist. Tried: {[cfg['path']] + alt_paths}")
    
    print(f"[DEBUG] Loading model {model_key} from {model_path}")
    
    # Check if this is a seq2seq (translation) model
    model_type = cfg.get("type", "causal")
    
    # Use appropriate dtype and device for GPU
    if DEVICE == "cuda":
        # Use float16 for GPU to save memory and speed up inference
        torch_dtype = torch.float16
        device_map = "cuda:0"
        print(f"[DEBUG] Loading on GPU with float16")
    else:
        torch_dtype = torch.float32
        device_map = "cpu"
        print(f"[DEBUG] Loading on CPU with float32")
    
    # Load appropriate model type
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )
        print(f"[DEBUG] Loaded seq2seq model for translation")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )
        print(f"[DEBUG] Loaded causal LM model")
    
    model.eval()
    print(f"[DEBUG] Model loaded successfully on {DEVICE}")
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


def _run_translation(req: TranslateRequest) -> TranslateResponse:
    """Run translation using a seq2seq model."""
    print(f"[DEBUG] Loading tokenizer for {req.model}")
    tokenizer = load_tokenizer(req.model)
    print(f"[DEBUG] Loading model for {req.model}")
    model = load_model(req.model)
    
    cfg = MODEL_REGISTRY.get(req.model)
    if not cfg:
        raise ValueError(f"unknown model: {req.model}")
    
    # Get source and target languages from request or model config
    source_lang = req.source_lang or cfg.get("src_lang", "eng_Latn")
    target_lang = req.target_lang or cfg.get("tgt_lang", "arb_Arab")
    
    print(f"[DEBUG] Translating from {source_lang} to {target_lang}")
    print(f"[DEBUG] Input text: {req.text[:50]}...")
    
    # Handle NLLB-200 models
    if "nllb" in req.model.lower():
        # NLLB models require setting src_lang attribute
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = source_lang
        else:
            print(f"[WARNING] Tokenizer doesn't have src_lang attribute, using default")
        
        # Tokenize with source language
        inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Get forced_bos_token_id for target language
        forced_bos_token_id = None
        if hasattr(tokenizer, "lang_code_to_id") and target_lang in tokenizer.lang_code_to_id:
            forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
            print(f"[DEBUG] Using forced_bos_token_id: {forced_bos_token_id} for {target_lang}")
        elif hasattr(tokenizer, "convert_tokens_to_ids"):
            # Fallback: try to convert target_lang directly
            try:
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
            except:
                pass
        
        # Generate translation
        with torch.no_grad():
            generation_kwargs = {
                "max_length": req.max_length,
                "num_beams": 5,  # Beam search for better quality
                "early_stopping": True,
            }
            
            if forced_bos_token_id is not None:
                generation_kwargs["forced_bos_token_id"] = forced_bos_token_id
            
            output_ids = model.generate(**inputs, **generation_kwargs)
    
    # Handle M2M100 models
    elif "m2m100" in req.model.lower():
        # M2M100 models use different language codes (e.g., "en", "ar")
        tokenizer.src_lang = source_lang
        
        # Tokenize input
        inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate with target language
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                max_length=req.max_length,
                num_beams=5,
                early_stopping=True
            )
    
    # Generic seq2seq model handling
    else:
        # Tokenize input
        inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=req.max_length,
                num_beams=5,
                early_stopping=True
            )
    
    # Decode translation
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    token_count = len(tokenizer.tokenize(translated_text))
    
    print(f"[DEBUG] Translation complete: {token_count} tokens")
    return TranslateResponse(
        translated_text=translated_text.strip(),
        source_lang=source_lang,
        target_lang=target_lang,
        tokens=token_count
    )


@app.post("/v1/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    """Translate text using a seq2seq model.
    
    Supports both direct translation and TOON-enhanced translation.
    TOON can be enabled via use_toon parameter.
    """
    try:
        cfg = MODEL_REGISTRY.get(req.model)
        if not cfg:
            raise ValueError(f"unknown model: {req.model}")
        
        model_type = cfg.get("type", "causal")
        if model_type != "seq2seq":
            raise ValueError(f"model {req.model} is not a translation model (type: {model_type})")
        
        # Check if TOON is enabled (via request or environment)
        # Default to True (TOON enabled by default for better quality)
        # Can be disabled by setting use_toon=False or ENABLE_TOON=false
        if req.use_toon is not None:
            enable_toon = req.use_toon
        else:
            # Check environment variable, default to True
            enable_toon = os.getenv("ENABLE_TOON", "true").lower() == "true"
        fallback_on_error = os.getenv("TOON_FALLBACK_ON_ERROR", "true").lower() == "true"
        
        if enable_toon:
            try:
                # Import TOON modules
                from toon_translation import translate_with_toon
                
                # Get source and target languages
                source_lang = req.source_lang or cfg.get("src_lang", "eng_Latn")
                target_lang = req.target_lang or cfg.get("tgt_lang", "arb_Arab")
                
                # Check cache first
                cache_enabled = os.getenv("TOON_CACHE_ENABLED", "true").lower() == "true"
                cached_result = None
                cache = None
                if cache_enabled:
                    try:
                        from toon_cache import get_cache
                        cache = get_cache()
                        if cache:
                            cached_result = cache.get(
                                text=req.text,
                                source_lang=source_lang,
                                target_lang=target_lang,
                                model=req.model,
                                use_toon=True,
                                toon_config=req.toon_config
                            )
                            if cached_result:
                                logger.info("Cache hit for translation request")
                                translated_text, toon_data_dict = cached_result
                                response = TranslateResponse(
                                    translated_text=translated_text,
                                    source_lang=source_lang,
                                    target_lang=target_lang,
                                    tokens=len(translated_text.split()),
                                    toon_used=True,
                                    toon_data=toon_data_dict
                                )
                                if METRICS_ENABLED and metrics and request_id:
                                    metrics.complete_request(request_id, response.tokens, success=True)
                                return response
                    except Exception as cache_error:
                        logger.warning(f"Cache lookup failed: {cache_error}")
                
                # Prepare TOON config
                toon_config = req.toon_config or {}
                models_base = os.getenv("TOON_MODELS_PATH") or os.getenv("MODELS_BASE", "/models")
                toon_config.setdefault("models_base", models_base)
                
                # Run TOON-enhanced translation
                translated_text, bilingual_toon = translate_with_toon(
                    text=req.text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    translation_model_name=req.model,
                    load_tokenizer=load_tokenizer,
                    load_model=load_model,
                    device=DEVICE,
                    max_length=req.max_length,
                    toon_config=toon_config,
                    fallback_on_error=fallback_on_error
                )
                
                # Prepare response
                token_count = len(translated_text.split())  # Approximate token count
                toon_data_dict = bilingual_toon.model_dump(mode='json') if bilingual_toon else None
                
                # Cache the result
                if cache_enabled and cache and not cached_result:
                    try:
                        cache.set(
                            text=req.text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            model=req.model,
                            use_toon=True,
                            translated_text=translated_text,
                            toon_data=toon_data_dict,
                            toon_config=req.toon_config
                        )
                    except Exception as cache_error:
                        logger.warning(f"Cache store failed: {cache_error}")
                
                response = TranslateResponse(
                    translated_text=translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    tokens=token_count,
                    toon_used=True,
                    toon_data=toon_data_dict
                )
                
                # Record metrics
                if METRICS_ENABLED and metrics and request_id:
                    metrics.complete_request(request_id, token_count, success=True)
                
                return response
                
            except Exception as toon_error:
                logger.warning(f"TOON translation failed: {toon_error}")
                if fallback_on_error:
                    logger.info("Falling back to direct translation...")
                    # Fall through to direct translation
                else:
                    if METRICS_ENABLED and metrics and request_id:
                        metrics.complete_request(request_id, 0, success=False, error=str(toon_error))
                    raise HTTPException(status_code=500, detail=f"TOON translation failed: {toon_error}")
        
        # Direct translation (default or fallback)
        response = _run_translation(req)
        response.toon_used = False
        response.toon_data = None
        
        # Record metrics for direct translation
        if METRICS_ENABLED and metrics and request_id:
            metrics.complete_request(request_id, response.tokens, success=True)
        
        return response
        
    except ValueError as exc:
        if METRICS_ENABLED and metrics and request_id:
            metrics.complete_request(request_id, 0, success=False, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Translation failed: {exc}", exc_info=True)
        if METRICS_ENABLED and metrics and request_id:
            metrics.complete_request(request_id, 0, success=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(exc)}") from exc


class TOONTranslateRequest(BaseModel):
    """Request for TOON-specific translation endpoint."""
    model: str
    text: str
    source_lang: str | None = None
    target_lang: str | None = None
    max_length: int = 512
    toon_config: Dict[str, Any] | None = None
    return_toon: bool = True  # Whether to return TOON representation


class TOONTranslateResponse(BaseModel):
    """Response from TOON-specific translation endpoint."""
    translated_text: str
    source_lang: str
    target_lang: str
    tokens: int
    toon_document: Dict[str, Any] | None = None  # Arabic TOON document
    bilingual_toon: Dict[str, Any] | None = None  # Bilingual TOON with alignment


class BatchTranslateRequest(BaseModel):
    """Request for batch translation."""
    model: str
    texts: List[str]
    source_lang: str | None = None
    target_lang: str | None = None
    max_length: int = 512
    use_toon: bool = True
    toon_config: Dict[str, Any] | None = None
    max_workers: int = 4
    batch_size: int = 10


class BatchTranslateResponse(BaseModel):
    """Response from batch translation endpoint."""
    results: List[Dict[str, Any]]
    total: int
    successful: int
    failed: int
    total_duration_ms: float
    avg_duration_ms: float


@app.post("/v1/translate/toon", response_model=TOONTranslateResponse)
def translate_toon(req: TOONTranslateRequest) -> TOONTranslateResponse:
    """Translate text using TOON pipeline and return TOON representation.
    
    This endpoint is useful for debugging and inspecting the TOON representation.
    """
    try:
        cfg = MODEL_REGISTRY.get(req.model)
        if not cfg:
            raise ValueError(f"unknown model: {req.model}")
        
        model_type = cfg.get("type", "causal")
        if model_type != "seq2seq":
            raise ValueError(f"model {req.model} is not a translation model (type: {model_type})")
        
        # Import TOON modules
        from toon_translation import translate_with_toon
        from toon_generator import ArabicTOONGenerator
        
        # Get source and target languages
        source_lang = req.source_lang or cfg.get("src_lang", "eng_Latn")
        target_lang = req.target_lang or cfg.get("tgt_lang", "arb_Arab")
        
        # Prepare TOON config
        toon_config = req.toon_config or {}
        models_base = os.getenv("TOON_MODELS_PATH") or os.getenv("MODELS_BASE", "/models")
        toon_config.setdefault("models_base", models_base)
        
        # Generate Arabic TOON
        toon_generator = ArabicTOONGenerator(
            models_base=toon_config.get("models_base"),
            use_arbml=toon_config.get("use_arbml", True),
            use_qadi=toon_config.get("use_qadi", False),
            use_kuwain=toon_config.get("use_kuwain", False)
        )
        arabic_toon = toon_generator.generate_toon(req.text, language=source_lang)
        
        # Run TOON-enhanced translation
        translated_text, bilingual_toon = translate_with_toon(
            text=req.text,
            source_lang=source_lang,
            target_lang=target_lang,
            translation_model_name=req.model,
            load_tokenizer=load_tokenizer,
            load_model=load_model,
            device=DEVICE,
            max_length=req.max_length,
            toon_config=toon_config,
            fallback_on_error=False  # Don't fallback in TOON-specific endpoint
        )
        
        # Prepare response
        token_count = len(translated_text.split())
        
        response = TOONTranslateResponse(
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            tokens=token_count,
            toon_document=arabic_toon.model_dump(mode='json') if req.return_toon else None,
            bilingual_toon=bilingual_toon.model_dump(mode='json') if (req.return_toon and bilingual_toon) else None
        )
        
        # Record metrics
        if METRICS_ENABLED and metrics and request_id:
            metrics.complete_request(request_id, token_count, success=True)
        
        return response
        
    except ValueError as exc:
        if METRICS_ENABLED and metrics and request_id:
            metrics.complete_request(request_id, 0, success=False, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"TOON translation failed: {exc}", exc_info=True)
        if METRICS_ENABLED and metrics and request_id:
            metrics.complete_request(request_id, 0, success=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"TOON translation failed: {str(exc)}") from exc


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "models": list(MODEL_REGISTRY.keys())}


@app.get("/metrics")
def get_metrics():
    """Get translation metrics."""
    if not METRICS_ENABLED or not metrics:
        return JSONResponse(
            status_code=503,
            content={"error": "Metrics not available"}
        )
    
    stats = metrics.get_stats()
    recent = metrics.get_recent_requests(limit=20)
    
    return {
        "stats": stats,
        "recent_requests": recent,
        "timestamp": time.time()
    }


@app.get("/metrics/stats")
def get_stats():
    """Get aggregated statistics."""
    if not METRICS_ENABLED or not metrics:
        return JSONResponse(
            status_code=503,
            content={"error": "Metrics not available"}
        )
    
    stats = metrics.get_stats()
    
    # Add cache stats if available
    try:
        from toon_cache import get_cache
        cache = get_cache()
        if cache:
            stats["cache"] = cache.get_stats()
    except Exception:
        pass
    
    return stats


@app.post("/cache/clear")
def clear_cache():
    """Clear the translation cache."""
    try:
        from toon_cache import get_cache
        cache = get_cache()
        if cache:
            cache.clear()
            return {"status": "success", "message": "Cache cleared"}
        else:
            return JSONResponse(
                status_code=503,
                content={"error": "Cache not available"}
            )
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/v1/translate/batch", response_model=BatchTranslateResponse)
def translate_batch_endpoint(req: BatchTranslateRequest) -> BatchTranslateResponse:
    """Translate a batch of texts.
    
    Processes multiple texts in parallel for improved throughput.
    """
    request_id = None
    if METRICS_ENABLED and metrics:
        # Track batch request
        request_id = metrics.start_request(
            model=req.model,
            source_lang=req.source_lang or "unknown",
            target_lang=req.target_lang or "unknown",
            text_length=sum(len(t) for t in req.texts),
            use_toon=req.use_toon
        )
        logger.info(f"Batch translation request started: {request_id} ({len(req.texts)} texts)")
    
    try:
        cfg = MODEL_REGISTRY.get(req.model)
        if not cfg:
            raise ValueError(f"unknown model: {req.model}")
        
        model_type = cfg.get("type", "causal")
        if model_type != "seq2seq":
            raise ValueError(f"model {req.model} is not a translation model (type: {model_type})")
        
        # Get source and target languages
        source_lang = req.source_lang or cfg.get("src_lang", "eng_Latn")
        target_lang = req.target_lang or cfg.get("tgt_lang", "arb_Arab")
        
        # Prepare TOON config
        toon_config = req.toon_config or {}
        models_base = os.getenv("TOON_MODELS_PATH") or os.getenv("MODELS_BASE", "/models")
        toon_config.setdefault("models_base", models_base)
        
        # Import batch translation
        from batch_translation import translate_batch
        
        start_time = time.time()
        
        # Process batch
        results = translate_batch(
            texts=req.texts,
            source_lang=source_lang,
            target_lang=target_lang,
            translation_model_name=req.model,
            load_tokenizer=load_tokenizer,
            load_model=load_model,
            device=DEVICE,
            max_length=req.max_length,
            use_toon=req.use_toon,
            toon_config=toon_config,
            max_workers=req.max_workers,
            batch_size=req.batch_size
        )
        
        total_duration_ms = (time.time() - start_time) * 1000
        
        # Calculate statistics
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        avg_duration_ms = total_duration_ms / len(results) if results else 0.0
        
        # Record metrics
        if METRICS_ENABLED and metrics and request_id:
            total_tokens = sum(len(r.get("translated_text", "").split()) for r in results)
            metrics.complete_request(request_id, total_tokens, success=(failed == 0))
        
        response = BatchTranslateResponse(
            results=results,
            total=len(results),
            successful=successful,
            failed=failed,
            total_duration_ms=total_duration_ms,
            avg_duration_ms=avg_duration_ms
        )
        
        logger.info(
            f"Batch translation complete: {successful}/{len(results)} successful, "
            f"duration: {total_duration_ms:.2f}ms"
        )
        
        return response
        
    except ValueError as exc:
        if METRICS_ENABLED and metrics and request_id:
            metrics.complete_request(request_id, 0, success=False, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Batch translation failed: {exc}", exc_info=True)
        if METRICS_ENABLED and metrics and request_id:
            metrics.complete_request(request_id, 0, success=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(exc)}") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TRANSFORMERS_CPU_PORT", "9090"))
    uvicorn.run("transformers_cpu_server:app", host="0.0.0.0", port=port, reload=False)
