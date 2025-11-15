"""TOON (Token Oriented Object Notation) schema definitions.

TOON provides a structured intermediate representation for Arabic-to-English
translation, making linguistic features explicit rather than implicit in embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class Dialect(str, Enum):
    """Arabic dialect classification."""
    MSA = "MSA"  # Modern Standard Arabic
    EGYPTIAN = "EGYPTIAN"
    LEVANTINE = "LEVANTINE"
    GULF = "GULF"
    MAGHREBI = "MAGHREBI"
    UNKNOWN = "UNKNOWN"


class POS(str, Enum):
    """Part-of-speech tags."""
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"
    ADV = "ADV"
    PRON = "PRON"
    PREP = "PREP"
    CONJ = "CONJ"
    PART = "PART"
    INTERJ = "INTERJ"
    UNKNOWN = "UNKNOWN"


class TOONMorphology(BaseModel):
    """Morphological features of a token."""
    gender: Optional[str] = None  # masculine, feminine
    number: Optional[str] = None  # singular, dual, plural
    person: Optional[str] = None  # first, second, third
    case: Optional[str] = None  # nominative, accusative, genitive
    mood: Optional[str] = None  # indicative, subjunctive, jussive
    voice: Optional[str] = None  # active, passive
    definiteness: Optional[str] = None  # definite, indefinite
    additional: Dict[str, Any] = Field(default_factory=dict)  # Additional morphological features


class TOONToken(BaseModel):
    """A single token with its linguistic properties."""
    token: str = Field(..., description="The surface form of the token")
    lemma: Optional[str] = Field(None, description="The lemma/base form")
    pos: Optional[POS] = Field(None, description="Part of speech tag")
    morphology: Optional[TOONMorphology] = Field(None, description="Morphological features")
    embedding: Optional[List[float]] = Field(None, description="Token embedding vector")
    relations: List[str] = Field(default_factory=list, description="Syntactic/semantic relations (e.g., 'subject', 'object', 'root')")
    dialect: Optional[Dialect] = Field(None, description="Dialect classification")
    position: int = Field(..., description="Position in the document")
    confidence: Optional[float] = Field(None, description="Confidence score for linguistic analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TOONDocument(BaseModel):
    """A document represented as a collection of TOON tokens."""
    tokens: List[TOONToken] = Field(default_factory=list, description="List of TOON tokens")
    source_text: str = Field(..., description="Original source text")
    language: str = Field(default="arb_Arab", description="Source language code")
    dialect: Optional[Dialect] = Field(None, description="Overall document dialect")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document-level metadata")
    
    def get_tokens_by_pos(self, pos: POS) -> List[TOONToken]:
        """Get all tokens with a specific part of speech."""
        return [t for t in self.tokens if t.pos == pos]
    
    def get_tokens_by_relation(self, relation: str) -> List[TOONToken]:
        """Get all tokens with a specific relation."""
        return [t for t in self.tokens if relation in t.relations]
    
    def to_text(self) -> str:
        """Reconstruct text from tokens."""
        return " ".join(t.token for t in self.tokens)


class TOONAlignment(BaseModel):
    """Cross-lingual alignment between Arabic and English TOON objects."""
    arabic_token_idx: int = Field(..., description="Index of Arabic token in source document")
    english_token_idx: int = Field(..., description="Index of English token in target document")
    alignment_type: str = Field(default="word", description="Type of alignment (word, phrase, etc.)")
    confidence: float = Field(default=1.0, description="Alignment confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional alignment metadata")


class BilingualTOON(BaseModel):
    """Bilingual TOON representation with alignment."""
    source: TOONDocument = Field(..., description="Source language TOON document")
    target: TOONDocument = Field(..., description="Target language TOON document")
    alignments: List[TOONAlignment] = Field(default_factory=list, description="Cross-lingual alignments")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Bilingual TOON metadata")
    
    def get_aligned_tokens(self, arabic_token_idx: int) -> List[TOONToken]:
        """Get English tokens aligned to a specific Arabic token."""
        aligned_indices = [
            a.english_token_idx 
            for a in self.alignments 
            if a.arabic_token_idx == arabic_token_idx
        ]
        return [self.target.tokens[i] for i in aligned_indices if i < len(self.target.tokens)]
    
    def get_aligned_arabic_tokens(self, english_token_idx: int) -> List[TOONToken]:
        """Get Arabic tokens aligned to a specific English token."""
        aligned_indices = [
            a.arabic_token_idx 
            for a in self.alignments 
            if a.english_token_idx == english_token_idx
        ]
        return [self.source.tokens[i] for i in aligned_indices if i < len(self.source.tokens)]

