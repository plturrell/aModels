"""Performance optimization utilities for translation service."""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
import torch

logger = logging.getLogger(__name__)


def optimize_model_for_inference(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Optimize model for inference.
    
    Args:
        model: PyTorch model
        device: Device to use (cuda/cpu)
    
    Returns:
        Optimized model
    """
    model.eval()  # Set to evaluation mode
    
    if device == "cuda" and torch.cuda.is_available():
        # Enable optimizations for CUDA
        if hasattr(torch.backends.cudnn, "benchmark"):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "deterministic"):
            torch.backends.cudnn.deterministic = False
        
        # Move model to GPU
        model = model.to(device)
        
        # Enable half precision if supported
        if torch.cuda.is_available() and hasattr(model, "half"):
            try:
                model = model.half()
                logger.info("Model optimized with half precision")
            except Exception as e:
                logger.warning(f"Half precision not available: {e}")
    
    # Enable torch.jit optimization if possible
    try:
        if hasattr(torch.jit, "optimize_for_inference"):
            model = torch.jit.optimize_for_inference(model)
            logger.info("Model optimized with torch.jit")
    except Exception as e:
        logger.debug(f"torch.jit optimization not available: {e}")
    
    return model


def batch_tokenize(
    texts: list[str],
    tokenizer: Any,
    max_length: int = 512,
    padding: bool = True,
    truncation: bool = True
) -> Dict[str, torch.Tensor]:
    """Efficiently tokenize a batch of texts.
    
    Args:
        texts: List of texts to tokenize
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
    
    Returns:
        Dictionary of tokenized inputs
    """
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=padding,
        truncation=truncation,
        max_length=max_length
    )


def optimize_generation_config(
    max_length: int = 512,
    num_beams: int = 5,
    early_stopping: bool = True,
    use_cache: bool = True,
    do_sample: bool = False
) -> Dict[str, Any]:
    """Create optimized generation configuration.
    
    Args:
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
        early_stopping: Whether to stop early
        use_cache: Whether to use KV cache
        do_sample: Whether to use sampling
    
    Returns:
        Generation configuration dictionary
    """
    return {
        "max_length": max_length,
        "num_beams": num_beams,
        "early_stopping": early_stopping,
        "use_cache": use_cache,
        "do_sample": do_sample,
        "pad_token_id": None,  # Will be set by tokenizer
        "eos_token_id": None,  # Will be set by tokenizer
    }


def measure_performance(func: Callable) -> Callable:
    """Decorator to measure function performance.
    
    Args:
        func: Function to measure
    
    Returns:
        Wrapped function with performance measurement
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = None
        
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            end_memory = None
            
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                memory_used = (end_memory - start_memory) / 1024**2  # MB
                logger.debug(
                    f"{func.__name__}: {duration:.3f}s, "
                    f"GPU memory: {memory_used:.2f}MB"
                )
            else:
                logger.debug(f"{func.__name__}: {duration:.3f}s")
    
    return wrapper


class ModelCache:
    """Cache for loaded models to avoid reloading."""
    
    def __init__(self, max_size: int = 5):
        """Initialize model cache.
        
        Args:
            max_size: Maximum number of models to cache
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model if cached, None otherwise
        """
        if model_name in self.cache:
            self.access_times[model_name] = time.time()
            return self.cache[model_name]
        return None
    
    def put(self, model_name: str, model: Any):
        """Put model in cache.
        
        Args:
            model_name: Name of the model
            model: Model to cache
        """
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size and model_name not in self.cache:
            oldest = min(self.access_times.items(), key=lambda x: x[1])
            del self.cache[oldest[0]]
            del self.access_times[oldest[0]]
        
        self.cache[model_name] = model
        self.access_times[model_name] = time.time()
        logger.debug(f"Cached model: {model_name}")
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Model cache cleared")


# Global model cache
_model_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get the global model cache."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache

