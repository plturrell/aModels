"""Arabic TOON generator for Stage 1: Arabic comprehension and TOON generation."""

from __future__ import annotations

import os
import re
from typing import Optional, List, Dict, Any
import logging

from toon_schema import (
    TOONToken,
    TOONDocument,
    TOONMorphology,
    Dialect,
    POS
)

# Import QADI integration for country-to-dialect mapping
try:
    from qadi_integration import map_country_to_dialect, QADIDataset
    HAS_QADI_INTEGRATION = True
except ImportError:
    HAS_QADI_INTEGRATION = False
    logger = logging.getLogger(__name__)
    logger.debug("QADI integration module not available")

logger = logging.getLogger(__name__)

# Try to import ARBML tools (camel-tools)
try:
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_teh_marbuta_ar
    HAS_ARBML = True
except ImportError:
    HAS_ARBML = False
    logger.warning("camel-tools not available, ARBML features will be limited")

# Try to import QADI for dialect classification
# QADI (QCRI Arabic Dialect Identification) can be integrated via:
# 1. camel-tools dialect identification (DIDPredictor)
# 2. Standalone QADI model from HuggingFace
# 3. Custom dialect classifier
try:
    # Try camel-tools dialect identification (DIDPred is the correct class)
    from camel_tools.dialectid import DIDPred
    HAS_QADI_CAMEL = True
except ImportError:
    try:
        # Fallback to alternative names
        from camel_tools.dialectid import DialectIdentifier as DIDPred
        HAS_QADI_CAMEL = True
    except ImportError:
        HAS_QADI_CAMEL = False
        DIDPred = None

# Try standalone QADI model
try:
    from transformers import pipeline
    HAS_QADI_TRANSFORMERS = True
except ImportError:
    HAS_QADI_TRANSFORMERS = False

HAS_QADI = HAS_QADI_CAMEL or HAS_QADI_TRANSFORMERS

# Try to import Kuwain model loader
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ArabicTOONGenerator:
    """Generate TOON representation from Arabic text."""
    
    def __init__(
        self,
        models_base: Optional[str] = None,
        use_arbml: bool = True,
        use_qadi: bool = False,
        use_kuwain: bool = False
    ):
        """Initialize the TOON generator.
        
        Args:
            models_base: Base path to models directory
            use_arbml: Whether to use ARBML tools for morphological analysis
            use_qadi: Whether to use QADI for dialect classification
            use_kuwain: Whether to use Kuwain model for enhanced analysis
        """
        self.models_base = models_base or os.getenv("MODELS_BASE", "/models")
        self.use_arbml = use_arbml and HAS_ARBML
        self.use_qadi = use_qadi and HAS_QADI
        self.use_kuwain = use_kuwain and HAS_TRANSFORMERS
        
        # Initialize ARBML components if available
        self.analyzer = None
        if self.use_arbml:
            try:
                # Load ARBML morphology database
                # camel-tools provides a built-in database
                db = MorphologyDB.builtin_db()
                self.analyzer = Analyzer(db)
                logger.info("✅ ARBML analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ARBML analyzer: {e}")
                logger.info("ARBML features will be disabled, falling back to basic analysis")
                self.use_arbml = False
        
        # Initialize QADI if available
        self.qadi_predictor = None
        self.qadi_pipeline = None
        if self.use_qadi:
            try:
                # Try camel-tools dialect identification first
                if HAS_QADI_CAMEL and DIDPred:
                    try:
                        # DIDPred needs a model - try DIDModel26
                        from camel_tools.dialectid.model26 import DIDModel26
                        qadi_model = DIDModel26.pretrained()
                        self.qadi_predictor = DIDPred(qadi_model)
                        logger.info("✅ QADI (camel-tools DIDPred) initialized successfully")
                    except Exception as e:
                        logger.debug(f"QADI Model26 failed: {e}, trying alternative...")
                        try:
                            # Try Model6 as fallback
                            from camel_tools.dialectid.model6 import DIDModel6
                            qadi_model = DIDModel6.pretrained()
                            self.qadi_predictor = DIDPred(qadi_model)
                            logger.info("✅ QADI (camel-tools DIDPred Model6) initialized")
                        except Exception as e2:
                            logger.warning(f"Failed to load camel-tools QADI: {e2}")
                            self.qadi_predictor = None
                
                # Fallback to HuggingFace QADI model if available
                if not self.qadi_predictor and HAS_QADI_TRANSFORMERS:
                    try:
                        qadi_path = os.path.join(self.models_base, "arabic_models", "qadi")
                        if os.path.exists(qadi_path):
                            from transformers import pipeline
                            self.qadi_pipeline = pipeline(
                                "text-classification",
                                model=qadi_path,
                                device=-1  # Use CPU by default
                            )
                            logger.info("✅ QADI (HuggingFace) initialized successfully")
                        else:
                            logger.debug("QADI model not found at expected path, dialect classification will be limited")
                    except Exception as e:
                        logger.warning(f"Failed to load HuggingFace QADI: {e}")
                
                if not self.qadi_predictor and not self.qadi_pipeline:
                    logger.warning("QADI not available, dialect classification will use heuristics")
                    self.use_qadi = False
            except Exception as e:
                logger.warning(f"Failed to initialize QADI: {e}")
                self.use_qadi = False
        
        # Initialize Kuwain if available
        self.kuwain_model = None
        self.kuwain_tokenizer = None
        if self.use_kuwain:
            try:
                kuwain_path = os.path.join(self.models_base, "arabic_models", "kuwain-1.5B")
                if os.path.exists(kuwain_path):
                    self.kuwain_tokenizer = AutoTokenizer.from_pretrained(kuwain_path)
                    self.kuwain_model = AutoModelForCausalLM.from_pretrained(kuwain_path)
                    logger.info("Kuwain model loaded")
                else:
                    logger.warning("Kuwain model not found, disabling")
                    self.use_kuwain = False
            except Exception as e:
                logger.warning(f"Failed to load Kuwain model: {e}")
                self.use_kuwain = False
    
    def generate_toon(self, text: str, language: str = "arb_Arab") -> TOONDocument:
        """Generate TOON representation from Arabic text.
        
        Args:
            text: Input Arabic text
            language: Language code (default: arb_Arab)
            
        Returns:
            TOONDocument with analyzed tokens
        """
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Tokenize
        tokens = self._tokenize(normalized_text)
        
        # Generate TOON tokens
        toon_tokens = []
        for i, token_text in enumerate(tokens):
            toon_token = self._analyze_token(token_text, i, normalized_text)
            toon_tokens.append(toon_token)
        
        # Classify document dialect
        dialect = self._classify_dialect(normalized_text)
        
        # Create TOON document
        doc = TOONDocument(
            tokens=toon_tokens,
            source_text=text,
            language=language,
            dialect=dialect
        )
        
        return doc
    
    def _normalize_text(self, text: str) -> str:
        """Normalize Arabic text."""
        if self.use_arbml:
            try:
                # Use ARBML normalization functions
                text = normalize_alef_maksura_ar(text)
                text = normalize_teh_marbuta_ar(text)
            except Exception as e:
                logger.warning(f"ARBML normalization failed: {e}")
        
        # Basic normalization: remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize Arabic text."""
        if self.use_arbml:
            try:
                # Use ARBML word tokenizer
                tokens = simple_word_tokenize(text)
                return tokens
            except Exception as e:
                logger.warning(f"ARBML tokenization failed: {e}")
        
        # Fallback: simple whitespace tokenization
        tokens = text.split()
        return tokens
    
    def _analyze_token(
        self,
        token: str,
        position: int,
        context: str
    ) -> TOONToken:
        """Analyze a single token and create TOONToken."""
        # Basic analysis
        lemma = token  # Default: token is its own lemma
        pos = POS.UNKNOWN
        morphology = None
        relations = []
        dialect = None
        
        # Morphological analysis using ARBML
        if self.use_arbml and self.analyzer:
            try:
                analyses = self.analyzer.analyze(token)
                if analyses:
                    # Use first analysis (most likely)
                    analysis = analyses[0]
                    lemma = analysis.get('lex', token)
                    
                    # Extract POS
                    pos_tag = analysis.get('pos', '')
                    pos = self._map_pos_tag(pos_tag)
                    
                    # Extract morphology
                    morphology = self._extract_morphology(analysis)
            except Exception as e:
                logger.debug(f"Morphological analysis failed for '{token}': {e}")
        
        # Extract relations (simplified - would use dependency parsing in full implementation)
        relations = self._extract_relations(token, position, context)
        
        # Classify token dialect if QADI is available
        # Note: Per-token dialect classification is simplified
        # Document-level dialect is set in generate_toon()
        dialect = None
        
        # Generate embedding if Kuwain is available
        embedding = None
        if self.use_kuwain and self.kuwain_tokenizer and self.kuwain_model:
            try:
                import torch
                # Get token embedding from Kuwain
                inputs = self.kuwain_tokenizer(token, return_tensors="pt")
                # Move to same device as model
                device = next(self.kuwain_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.kuwain_model(**inputs, output_hidden_states=True)
                    # Use last hidden state, average over sequence length
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        hidden_state = outputs.hidden_states[-1]
                        embedding = hidden_state.mean(dim=1).squeeze().cpu().tolist()
                    elif hasattr(outputs, 'last_hidden_state'):
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
            except Exception as e:
                logger.debug(f"Embedding generation failed for '{token}': {e}")
        
        return TOONToken(
            token=token,
            lemma=lemma,
            pos=pos,
            morphology=morphology,
            embedding=embedding,
            relations=relations,
            dialect=dialect,
            position=position
        )
    
    def _map_pos_tag(self, pos_tag: str) -> POS:
        """Map ARBML POS tag to TOON POS enum."""
        pos_mapping = {
            'noun': POS.NOUN,
            'verb': POS.VERB,
            'adj': POS.ADJ,
            'adv': POS.ADV,
            'pron': POS.PRON,
            'prep': POS.PREP,
            'conj': POS.CONJ,
            'part': POS.PART,
            'interj': POS.INTERJ,
        }
        pos_lower = pos_tag.lower()
        return pos_mapping.get(pos_lower, POS.UNKNOWN)
    
    def _extract_morphology(self, analysis: Dict[str, Any]) -> TOONMorphology:
        """Extract morphological features from ARBML analysis."""
        morph = TOONMorphology()
        
        # Extract gender
        if 'gen' in analysis:
            morph.gender = analysis['gen']
        
        # Extract number
        if 'num' in analysis:
            morph.number = analysis['num']
        
        # Extract person
        if 'per' in analysis:
            morph.person = analysis['per']
        
        # Extract case
        if 'case' in analysis:
            morph.case = analysis['case']
        
        # Extract mood
        if 'mod' in analysis:
            morph.mood = analysis['mod']
        
        # Extract voice
        if 'vox' in analysis:
            morph.voice = analysis['vox']
        
        # Extract definiteness
        if 'def' in analysis:
            morph.definiteness = analysis['def']
        
        # Store additional features
        for key, value in analysis.items():
            if key not in ['lex', 'pos', 'gen', 'num', 'per', 'case', 'mod', 'vox', 'def']:
                morph.additional[key] = value
        
        return morph
    
    def _extract_relations(
        self,
        token: str,
        position: int,
        context: str
    ) -> List[str]:
        """Extract syntactic/semantic relations for a token.
        
        This is a simplified implementation. A full version would use
        dependency parsing or other NLP tools.
        """
        relations = []
        
        # Simple heuristics
        if position == 0:
            relations.append("first_token")
        
        # Check for common Arabic particles that indicate relations
        if token in ['في', 'من', 'إلى', 'على', 'عن', 'مع']:
            relations.append("preposition")
        
        if token in ['الذي', 'التي', 'اللذان', 'اللتان']:
            relations.append("relative_pronoun")
        
        return relations
    
    def _classify_dialect(self, text: str) -> Dialect:
        """Classify the dialect of the text."""
        if self.use_qadi:
            try:
                # Try camel-tools QADI first
                if self.qadi_predictor:
                    try:
                        predictions = self.qadi_predictor.predict([text])
                        if predictions and len(predictions) > 0:
                            # DIDPred returns list of strings or dicts
                            pred = predictions[0]
                            if isinstance(pred, dict):
                                dialect_label = pred.get('label', pred.get('dialect', 'MSA'))
                            else:
                                dialect_label = str(pred)
                            # Map camel-tools labels to our Dialect enum
                            dialect_map = {
                                'EGY': Dialect.EGYPTIAN,
                                'LEV': Dialect.LEVANTINE,
                                'GLF': Dialect.GULF,
                                'NOR': Dialect.MSA,  # North African -> MSA for now
                                'MSA': Dialect.MSA
                            }
                            return dialect_map.get(dialect_label.upper(), Dialect.MSA)
                    except Exception as e:
                        logger.debug(f"camel-tools QADI classification failed: {e}")
                
                # Try HuggingFace QADI pipeline
                if self.qadi_pipeline:
                    try:
                        result = self.qadi_pipeline(text)
                        if result and isinstance(result, list) and len(result) > 0:
                            label = result[0].get('label', 'MSA')
                            # Map to Dialect enum
                            dialect_map = {
                                'EGYPTIAN': Dialect.EGYPTIAN,
                                'LEVANTINE': Dialect.LEVANTINE,
                                'GULF': Dialect.GULF,
                                'MAGHREBI': Dialect.MAGHREBI,
                                'MSA': Dialect.MSA
                            }
                            return dialect_map.get(label.upper(), Dialect.MSA)
                    except Exception as e:
                        logger.debug(f"HuggingFace QADI classification failed: {e}")
            except Exception as e:
                logger.debug(f"QADI dialect classification failed: {e}")
        
        # Fallback: use heuristics or assume MSA
        # Simple heuristic: check for dialect-specific markers
        text_lower = text.lower()
        if any(marker in text_lower for marker in ['إيه', 'ازيك', 'عايز']):
            return Dialect.EGYPTIAN
        elif any(marker in text_lower for marker in ['شو', 'ليش', 'وين']):
            return Dialect.LEVANTINE
        elif any(marker in text_lower for marker in ['شلون', 'وين', 'شنو']):
            return Dialect.GULF
        
        return Dialect.MSA
    
    def _classify_token_dialect(self, token: str) -> Optional[Dialect]:
        """Classify the dialect of a single token."""
        # Use document-level dialect for now
        # Per-token dialect classification would require more sophisticated analysis
        # This could be enhanced with token-level QADI in the future
        return None

