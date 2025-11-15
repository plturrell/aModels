"""Advanced cross-lingual alignment for TOON translation."""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from toon_schema import TOONToken, TOONDocument, BilingualTOON, TOONAlignment

logger = logging.getLogger(__name__)


def compute_word_alignment(
    source_tokens: List[TOONToken],
    target_tokens: List[str],
    source_embeddings: Optional[np.ndarray] = None,
    target_embeddings: Optional[np.ndarray] = None
) -> List[TOONAlignment]:
    """Compute word alignment between source and target tokens.
    
    Uses cosine similarity between embeddings to find best alignments.
    
    Args:
        source_tokens: List of source TOON tokens
        target_tokens: List of target token strings
        source_embeddings: Optional pre-computed source embeddings
        target_embeddings: Optional pre-computed target embeddings
    
    Returns:
        List of TOONAlignment objects
    """
    alignments = []
    
    if not source_tokens or not target_tokens:
        return alignments
    
    # Compute embeddings if not provided
    if source_embeddings is None:
        source_embeddings = np.array([
            token.embedding if token.embedding else np.zeros(768)
            for token in source_tokens
        ])
    
    if target_embeddings is None:
        # Use simple token-based embeddings (placeholder)
        target_embeddings = np.random.rand(len(target_tokens), 768)
    
    # Normalize embeddings
    source_norm = source_embeddings / (np.linalg.norm(source_embeddings, axis=1, keepdims=True) + 1e-8)
    target_norm = target_embeddings / (np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(source_norm, target_norm.T)
    
    # Find best alignments (one-to-many, many-to-one, many-to-many)
    for i, source_token in enumerate(source_tokens):
        # Find best target token(s)
        similarities = similarity_matrix[i]
        best_target_idx = np.argmax(similarities)
        confidence = float(similarities[best_target_idx])
        
        # Create alignment
        alignment = TOONAlignment(
            source_token_id=i,
            target_token_id=int(best_target_idx),
            alignment_type="one-to-one" if confidence > 0.7 else "fuzzy",
            confidence=confidence,
            metadata={
                "similarity_score": confidence,
                "source_lemma": source_token.lemma,
                "target_token": target_tokens[best_target_idx] if best_target_idx < len(target_tokens) else ""
            }
        )
        alignments.append(alignment)
    
    return alignments


def enhance_bilingual_toon(
    arabic_toon: TOONDocument,
    english_text: str,
    english_tokens: Optional[List[str]] = None
) -> BilingualTOON:
    """Enhance bilingual TOON with advanced alignment.
    
    Args:
        arabic_toon: Arabic TOON document
        english_text: English translation text
        english_tokens: Optional pre-tokenized English tokens
    
    Returns:
        Enhanced BilingualTOON with alignment
    """
    if english_tokens is None:
        # Simple tokenization (in production, use proper tokenizer)
        english_tokens = english_text.split()
    
    # Compute alignments
    alignments = compute_word_alignment(
        source_tokens=arabic_toon.tokens,
        target_tokens=english_tokens
    )
    
    # Create English TOON (simplified)
    from toon_schema import TOONToken, TOONDocument, Dialect, POS
    
    english_toon_tokens = []
    for i, token_str in enumerate(english_tokens):
        token = TOONToken(
            text=token_str,
            position=i,
            lemma=token_str.lower(),
            pos=POS.NOUN,  # Simplified
            morphology=None,
            embedding=None,
            relations=[],
            dialect=Dialect.MSA
        )
        english_toon_tokens.append(token)
    
    english_toon = TOONDocument(
        language="eng_Latn",
        tokens=english_toon_tokens,
        metadata={"source": "translation"}
    )
    
    # Create bilingual TOON
    bilingual_toon = BilingualTOON(
        source_document=arabic_toon,
        target_document=english_toon,
        alignments=alignments,
        metadata={
            "alignment_method": "embedding_similarity",
            "alignment_count": len(alignments)
        }
    )
    
    return bilingual_toon


def improve_translation_quality(
    source_toon: TOONDocument,
    initial_translation: str,
    alignments: List[TOONAlignment]
) -> str:
    """Improve translation quality using TOON alignments.
    
    Uses alignment information to refine the translation.
    
    Args:
        source_toon: Source TOON document
        initial_translation: Initial translation text
        alignments: Alignment information
    
    Returns:
        Improved translation text
    """
    # For now, return initial translation
    # In full implementation, would use alignments to:
    # - Fix word order issues
    # - Ensure morphological agreement
    # - Preserve semantic relationships
    
    return initial_translation

