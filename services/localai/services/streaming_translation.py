"""Streaming translation support for real-time translation."""

from __future__ import annotations

import logging
from typing import Iterator, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


def translate_stream(
    text: str,
    source_lang: str,
    target_lang: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    max_length: int = 512,
    chunk_size: int = 50
) -> Iterator[Dict[str, Any]]:
    """Stream translation results as they are generated.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        tokenizer: Tokenizer instance
        model: Model instance
        device: Device to use (cuda/cpu)
        max_length: Maximum sequence length
        chunk_size: Size of chunks to process
    
    Yields:
        Dictionary with:
        - chunk: Translated chunk text
        - position: Position in the translation
        - done: Whether translation is complete
    """
    try:
        # Set source language
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = source_lang
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get forced_bos_token_id for target language
        forced_bos_token_id = None
        if hasattr(tokenizer, "lang_code_to_id") and target_lang in tokenizer.lang_code_to_id:
            forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
        
        # Generate with streaming
        with torch.no_grad():
            generation_kwargs = {
                "max_length": max_length,
                "num_beams": 5,
                "early_stopping": True,
            }
            if forced_bos_token_id is not None:
                generation_kwargs["forced_bos_token_id"] = forced_bos_token_id
            
            # Generate tokens one by one for streaming
            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode and stream results
        generated_ids = output_ids.sequences[0]
        full_text = ""
        
        # Stream in chunks
        for i in range(0, len(generated_ids), chunk_size):
            chunk_ids = generated_ids[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            
            # Only yield new text (not already sent)
            new_text = chunk_text[len(full_text):]
            if new_text:
                full_text = chunk_text
                yield {
                    "chunk": new_text,
                    "position": i,
                    "done": False
                }
        
        # Final chunk
        if full_text != chunk_text:
            yield {
                "chunk": chunk_text[len(full_text):],
                "position": len(generated_ids),
                "done": True
            }
        else:
            yield {
                "chunk": "",
                "position": len(generated_ids),
                "done": True
            }
    
    except Exception as e:
        logger.error(f"Streaming translation failed: {e}")
        yield {
            "chunk": "",
            "position": 0,
            "done": True,
            "error": str(e)
        }

