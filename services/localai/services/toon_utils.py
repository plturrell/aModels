"""TOON utilities for serialization, validation, and debugging."""

from __future__ import annotations

import json
from typing import Dict, Any, Optional
from pathlib import Path

from toon_schema import (
    TOONToken,
    TOONDocument,
    TOONAlignment,
    BilingualTOON
)


def serialize_toon_document(doc: TOONDocument) -> Dict[str, Any]:
    """Serialize a TOONDocument to a dictionary."""
    return doc.model_dump(mode='json')


def deserialize_toon_document(data: Dict[str, Any]) -> TOONDocument:
    """Deserialize a dictionary to a TOONDocument."""
    return TOONDocument(**data)


def serialize_bilingual_toon(bilingual: BilingualTOON) -> Dict[str, Any]:
    """Serialize a BilingualTOON to a dictionary."""
    return bilingual.model_dump(mode='json')


def deserialize_bilingual_toon(data: Dict[str, Any]) -> BilingualTOON:
    """Deserialize a dictionary to a BilingualTOON."""
    return BilingualTOON(**data)


def toon_document_to_json(doc: TOONDocument, indent: Optional[int] = None) -> str:
    """Convert a TOONDocument to JSON string."""
    return json.dumps(serialize_toon_document(doc), indent=indent, ensure_ascii=False)


def toon_document_from_json(json_str: str) -> TOONDocument:
    """Create a TOONDocument from JSON string."""
    data = json.loads(json_str)
    return deserialize_toon_document(data)


def bilingual_toon_to_json(bilingual: BilingualTOON, indent: Optional[int] = None) -> str:
    """Convert a BilingualTOON to JSON string."""
    return json.dumps(serialize_bilingual_toon(bilingual), indent=indent, ensure_ascii=False)


def bilingual_toon_from_json(json_str: str) -> BilingualTOON:
    """Create a BilingualTOON from JSON string."""
    data = json.loads(json_str)
    return deserialize_bilingual_toon(data)


def save_toon_document(doc: TOONDocument, filepath: str) -> None:
    """Save a TOONDocument to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(toon_document_to_json(doc, indent=2))


def load_toon_document(filepath: str) -> TOONDocument:
    """Load a TOONDocument from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return toon_document_from_json(f.read())


def save_bilingual_toon(bilingual: BilingualTOON, filepath: str) -> None:
    """Save a BilingualTOON to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(bilingual_toon_to_json(bilingual, indent=2))


def load_bilingual_toon(filepath: str) -> BilingualTOON:
    """Load a BilingualTOON from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return bilingual_toon_from_json(f.read())


def validate_toon_document(doc: TOONDocument) -> tuple[bool, Optional[str]]:
    """Validate a TOONDocument structure.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # Check that tokens have valid positions
        positions = [t.position for t in doc.tokens]
        if len(positions) != len(set(positions)):
            return False, "Duplicate token positions found"
        
        # Check that positions are sequential
        sorted_positions = sorted(positions)
        if sorted_positions != list(range(len(positions))):
            return False, "Token positions are not sequential"
        
        # Check that source text is not empty
        if not doc.source_text.strip():
            return False, "Source text is empty"
        
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def visualize_toon_document(doc: TOONDocument, max_tokens: Optional[int] = None) -> str:
    """Create a human-readable visualization of a TOONDocument."""
    lines = []
    lines.append(f"TOON Document: {doc.language}")
    lines.append(f"Dialect: {doc.dialect or 'Unknown'}")
    lines.append(f"Source Text: {doc.source_text[:100]}{'...' if len(doc.source_text) > 100 else ''}")
    lines.append(f"Tokens: {len(doc.tokens)}")
    lines.append("")
    lines.append("Token Analysis:")
    lines.append("-" * 80)
    
    tokens_to_show = doc.tokens[:max_tokens] if max_tokens else doc.tokens
    for i, token in enumerate(tokens_to_show):
        lines.append(f"[{i}] {token.token}")
        if token.lemma:
            lines.append(f"    Lemma: {token.lemma}")
        if token.pos:
            lines.append(f"    POS: {token.pos.value}")
        if token.morphology:
            morph_info = []
            if token.morphology.gender:
                morph_info.append(f"gender={token.morphology.gender}")
            if token.morphology.number:
                morph_info.append(f"number={token.morphology.number}")
            if morph_info:
                lines.append(f"    Morphology: {', '.join(morph_info)}")
        if token.relations:
            lines.append(f"    Relations: {', '.join(token.relations)}")
        if token.dialect:
            lines.append(f"    Dialect: {token.dialect.value}")
        lines.append("")
    
    if max_tokens and len(doc.tokens) > max_tokens:
        lines.append(f"... ({len(doc.tokens) - max_tokens} more tokens)")
    
    return "\n".join(lines)


def visualize_bilingual_toon(bilingual: BilingualTOON, max_tokens: Optional[int] = None) -> str:
    """Create a human-readable visualization of a BilingualTOON."""
    lines = []
    lines.append("Bilingual TOON")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Source (Arabic):")
    lines.append(visualize_toon_document(bilingual.source, max_tokens))
    lines.append("")
    lines.append("Target (English):")
    lines.append(visualize_toon_document(bilingual.target, max_tokens))
    lines.append("")
    lines.append(f"Alignments: {len(bilingual.alignments)}")
    if bilingual.alignments:
        lines.append("-" * 80)
        for alignment in bilingual.alignments[:max_tokens or 10]:
            src_token = bilingual.source.tokens[alignment.arabic_token_idx] if alignment.arabic_token_idx < len(bilingual.source.tokens) else None
            tgt_token = bilingual.target.tokens[alignment.english_token_idx] if alignment.english_token_idx < len(bilingual.target.tokens) else None
            if src_token and tgt_token:
                lines.append(f"  {src_token.token} ({alignment.arabic_token_idx}) <-> {tgt_token.token} ({alignment.english_token_idx}) [conf: {alignment.confidence:.2f}]")
    return "\n".join(lines)


def compare_toon_documents(doc1: TOONDocument, doc2: TOONDocument) -> Dict[str, Any]:
    """Compare two TOONDocuments and return differences."""
    differences = {
        "token_count_diff": len(doc1.tokens) - len(doc2.tokens),
        "pos_distribution_diff": {},
        "dialect_match": doc1.dialect == doc2.dialect,
        "common_tokens": [],
        "unique_to_doc1": [],
        "unique_to_doc2": []
    }
    
    # Compare POS distributions
    pos1 = {}
    pos2 = {}
    for token in doc1.tokens:
        if token.pos:
            pos1[token.pos.value] = pos1.get(token.pos.value, 0) + 1
    for token in doc2.tokens:
        if token.pos:
            pos2[token.pos.value] = pos2.get(token.pos.value, 0) + 1
    
    all_pos = set(pos1.keys()) | set(pos2.keys())
    for pos in all_pos:
        diff = pos1.get(pos, 0) - pos2.get(pos, 0)
        if diff != 0:
            differences["pos_distribution_diff"][pos] = diff
    
    # Find common and unique tokens
    tokens1 = {t.token for t in doc1.tokens}
    tokens2 = {t.token for t in doc2.tokens}
    differences["common_tokens"] = list(tokens1 & tokens2)
    differences["unique_to_doc1"] = list(tokens1 - tokens2)
    differences["unique_to_doc2"] = list(tokens2 - tokens1)
    
    return differences

