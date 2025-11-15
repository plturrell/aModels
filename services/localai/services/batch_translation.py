"""Batch translation support for processing multiple texts efficiently."""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


def translate_batch(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    translation_model_name: str,
    load_tokenizer,
    load_model,
    device: str,
    max_length: int = 512,
    use_toon: bool = True,
    toon_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 4,
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """Translate a batch of texts.
    
    Args:
        texts: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code
        translation_model_name: Name of the translation model
        load_tokenizer: Function to load tokenizer
        load_model: Function to load model
        device: Device to use (cuda/cpu)
        max_length: Maximum sequence length
        use_toon: Whether to use TOON enhancement
        toon_config: TOON configuration
        max_workers: Maximum number of parallel workers
        batch_size: Number of texts to process in each batch
    
    Returns:
        List of translation results, each containing:
        - text: Original text
        - translated_text: Translated text
        - success: Whether translation succeeded
        - error: Error message if failed
        - duration_ms: Translation duration in milliseconds
    """
    results = []
    
    # Load model and tokenizer once for the batch
    logger.info(f"Loading model {translation_model_name} for batch translation...")
    tokenizer = load_tokenizer(translation_model_name)
    model = load_model(translation_model_name)
    logger.info(f"Model loaded, processing {len(texts)} texts in batches of {batch_size}")
    
    # Process in batches
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
        
        # Process batch with parallel workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for text in batch_texts:
                future = executor.submit(
                    _translate_single,
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_length,
                    use_toon=use_toon,
                    toon_config=toon_config,
                    translation_model_name=translation_model_name,
                    load_tokenizer=load_tokenizer,
                    load_model=load_model
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch translation: {e}")
                    results.append({
                        "text": "",
                        "translated_text": "",
                        "success": False,
                        "error": str(e),
                        "duration_ms": 0
                    })
    
    logger.info(f"Batch translation complete: {len(results)} results")
    return results


def _translate_single(
    text: str,
    source_lang: str,
    target_lang: str,
    tokenizer,
    model,
    device: str,
    max_length: int,
    use_toon: bool,
    toon_config: Optional[Dict[str, Any]],
    translation_model_name: str,
    load_tokenizer,
    load_model
) -> Dict[str, Any]:
    """Translate a single text."""
    start_time = time.time()
    
    try:
        if use_toon:
            # Use TOON-enhanced translation
            from toon_translation import translate_with_toon
            
            translated_text, _ = translate_with_toon(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                translation_model_name=translation_model_name,
                load_tokenizer=lambda _: tokenizer,  # Return already loaded tokenizer
                load_model=lambda _: model,  # Return already loaded model
                device=device,
                max_length=max_length,
                toon_config=toon_config or {},
                fallback_on_error=True
            )
        else:
            # Direct translation
            import torch
            
            # Tokenize
            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = source_lang
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get forced_bos_token_id for target language
            forced_bos_token_id = None
            if hasattr(tokenizer, "lang_code_to_id") and target_lang in tokenizer.lang_code_to_id:
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
            
            # Generate
            with torch.no_grad():
                generation_kwargs = {
                    "max_length": max_length,
                    "num_beams": 5,
                    "early_stopping": True,
                }
                if forced_bos_token_id is not None:
                    generation_kwargs["forced_bos_token_id"] = forced_bos_token_id
                
                output_ids = model.generate(**inputs, **generation_kwargs)
            
            translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "text": text,
            "translated_text": translated_text,
            "success": True,
            "error": None,
            "duration_ms": duration_ms
        }
    
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Translation failed for text '{text[:50]}...': {e}")
        return {
            "text": text,
            "translated_text": "",
            "success": False,
            "error": str(e),
            "duration_ms": duration_ms
        }

