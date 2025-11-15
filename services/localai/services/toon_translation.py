"""TOON-enhanced translation pipeline."""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple, List

from toon_schema import (
    TOONDocument,
    BilingualTOON,
    TOONAlignment,
    Dialect
)
from toon_generator import ArabicTOONGenerator
from toon_utils import validate_toon_document

logger = logging.getLogger(__name__)


def translate_with_toon(
    text: str,
    source_lang: str,
    target_lang: str,
    translation_model_name: str,
    load_tokenizer,
    load_model,
    device: str,
    max_length: int = 512,
    toon_config: Optional[Dict[str, Any]] = None,
    fallback_on_error: bool = True
) -> Tuple[str, Optional[BilingualTOON]]:
    """Translate text using TOON-enhanced pipeline.
    
    Pipeline stages:
    1. Generate Arabic TOON from input text
    2. Cross-lingual alignment using NLLB-200
    3. Generate English from aligned TOON
    
    Args:
        text: Source text to translate
        source_lang: Source language code (e.g., 'arb_Arab')
        target_lang: Target language code (e.g., 'eng_Latn')
        translation_model_name: Name of translation model (e.g., 'nllb-200-1.3B')
        load_tokenizer: Function to load tokenizer
        load_model: Function to load model
        device: Device to run on ('cuda' or 'cpu')
        max_length: Maximum generation length
        toon_config: Optional TOON configuration
        fallback_on_error: Whether to fall back to direct translation on error
        
    Returns:
        Tuple of (translated_text, bilingual_toon or None)
    """
    toon_config = toon_config or {}
    
    try:
        # Stage 1: Generate Arabic TOON
        logger.info("Stage 1: Generating Arabic TOON...")
        toon_generator = ArabicTOONGenerator(
            models_base=toon_config.get("models_base"),
            use_arbml=toon_config.get("use_arbml", True),
            use_qadi=toon_config.get("use_qadi", False),
            use_kuwain=toon_config.get("use_kuwain", False)
        )
        
        arabic_toon = toon_generator.generate_toon(text, language=source_lang)
        
        # Validate TOON
        is_valid, error_msg = validate_toon_document(arabic_toon)
        if not is_valid:
            logger.warning(f"TOON validation failed: {error_msg}")
            if fallback_on_error:
                logger.info("Falling back to direct translation...")
                return _direct_translation_fallback(
                    text, source_lang, target_lang, translation_model_name,
                    load_tokenizer, load_model, device, max_length
                ), None
            else:
                raise ValueError(f"TOON validation failed: {error_msg}")
        
        logger.info(f"Generated TOON with {len(arabic_toon.tokens)} tokens")
        
        # Stage 2: Cross-lingual alignment using NLLB-200
        logger.info("Stage 2: Performing cross-lingual alignment...")
        tokenizer = load_tokenizer(translation_model_name)
        model = load_model(translation_model_name)
        
        # Use NLLB to translate (this provides the alignment target)
        # For now, we use direct translation and create alignment heuristically
        # In a full implementation, we would use word alignment tools
        
        # Translate using NLLB
        translated_text = _translate_with_nllb(
            text, source_lang, target_lang,
            tokenizer, model, device, max_length
        )
        
        # Stage 3: Generate English TOON and create alignment
        logger.info("Stage 3: Generating English TOON and alignment...")
        
        # Use advanced alignment if available
        try:
            from advanced_alignment import enhance_bilingual_toon
            bilingual_toon = enhance_bilingual_toon(
                arabic_toon=arabic_toon,
                english_text=translated_text
            )
            logger.info(f"Enhanced bilingual TOON with {len(bilingual_toon.alignments)} alignments")
        except Exception as e:
            logger.warning(f"Advanced alignment failed, using basic alignment: {e}")
            # Fallback to basic alignment
            bilingual_toon = None = _create_english_toon(translated_text, target_lang)
        
        # Create alignment (simplified - would use proper alignment tools)
        alignments = _create_alignments(arabic_toon, english_toon, text, translated_text)
        
        bilingual_toon = BilingualTOON(
            source=arabic_toon,
            target=english_toon,
            alignments=alignments
        )
        
        logger.info(f"Created bilingual TOON with {len(alignments)} alignments")
        
        return translated_text, bilingual_toon
        
    except Exception as e:
        logger.error(f"TOON translation failed: {e}", exc_info=True)
        if fallback_on_error:
            logger.info("Falling back to direct translation...")
            return _direct_translation_fallback(
                text, source_lang, target_lang, translation_model_name,
                load_tokenizer, load_model, device, max_length
            ), None
        else:
            raise


def _translate_with_nllb(
    text: str,
    source_lang: str,
    target_lang: str,
    tokenizer,
    model,
    device: str,
    max_length: int
) -> str:
    """Translate text using NLLB model."""
    import torch
    
    # Set source language
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get forced_bos_token_id for target language
    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id") and target_lang in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    
    # Generate translation
    with torch.no_grad():
        generation_kwargs = {
            "max_length": max_length,
            "num_beams": 5,
            "early_stopping": True,
        }
        
        if forced_bos_token_id is not None:
            generation_kwargs["forced_bos_token_id"] = forced_bos_token_id
        
        output_ids = model.generate(**inputs, **generation_kwargs)
    
    # Decode
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text.strip()


def _create_english_toon(text: str, language: str) -> TOONDocument:
    """Create a simple English TOON from translated text.
    
    This is a simplified version. In full implementation, would use
    English NLP tools (spaCy, NLTK, etc.) for proper analysis.
    """
    from toon_schema import TOONToken, TOONDocument, POS
    
    # Simple tokenization
    tokens = text.split()
    
    toon_tokens = []
    for i, token in enumerate(tokens):
        toon_token = TOONToken(
            token=token,
            lemma=token.lower(),  # Simplified
            pos=POS.UNKNOWN,  # Would use POS tagger in full implementation
            position=i
        )
        toon_tokens.append(toon_token)
    
    return TOONDocument(
        tokens=toon_tokens,
        source_text=text,
        language=language
    )


def _create_alignments(
    arabic_toon: TOONDocument,
    english_toon: TOONDocument,
    arabic_text: str,
    english_text: str
) -> List[TOONAlignment]:
    """Create alignments between Arabic and English TOON tokens.
    
    This is a simplified heuristic-based alignment.
    In full implementation, would use proper word alignment tools
    (e.g., fast_align, GIZA++, or neural alignment models).
    """
    alignments = []
    
    # Simple heuristic: align based on token position ratios
    # This is very basic - proper alignment would use statistical or neural methods
    arabic_len = len(arabic_toon.tokens)
    english_len = len(english_toon.tokens)
    
    if arabic_len == 0 or english_len == 0:
        return alignments
    
    # Create 1-to-1 alignments based on position ratio
    # This is a placeholder - real alignment would be more sophisticated
    for i in range(min(arabic_len, english_len)):
        alignment = TOONAlignment(
            arabic_token_idx=i,
            english_token_idx=i,
            alignment_type="word",
            confidence=0.8  # Placeholder confidence
        )
        alignments.append(alignment)
    
    return alignments


def _direct_translation_fallback(
    text: str,
    source_lang: str,
    target_lang: str,
    translation_model_name: str,
    load_tokenizer,
    load_model,
    device: str,
    max_length: int
) -> str:
    """Fallback to direct translation without TOON."""
    logger.info("Using direct translation (no TOON)")
    tokenizer = load_tokenizer(translation_model_name)
    model = load_model(translation_model_name)
    return _translate_with_nllb(text, source_lang, target_lang, tokenizer, model, device, max_length)

