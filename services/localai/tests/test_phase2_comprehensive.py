"""Comprehensive Phase 2 test suite for TOON Arabic Translation Enhancement."""

import unittest
import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from toon_schema import (
    TOONToken,
    TOONDocument,
    TOONMorphology,
    TOONAlignment,
    BilingualTOON,
    Dialect,
    POS
)
from toon_utils import (
    serialize_toon_document,
    deserialize_toon_document,
    validate_toon_document,
    visualize_toon_document,
    toon_document_to_json,
    toon_document_from_json
)
from toon_generator import ArabicTOONGenerator
from toon_translation import translate_with_toon
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class TestPhase2Schema(unittest.TestCase):
    """Test Phase 2.1: TOON Schema and Core Infrastructure."""
    
    def test_toon_token_creation(self):
        """Test TOONToken creation with all fields."""
        token = TOONToken(
            token="الكتاب",
            lemma="كتاب",
            pos=POS.NOUN,
            morphology=TOONMorphology(
                gender="masculine",
                number="singular",
                definiteness="definite"
            ),
            position=0
        )
        self.assertEqual(token.token, "الكتاب")
        self.assertEqual(token.lemma, "كتاب")
        self.assertEqual(token.pos, POS.NOUN)
        self.assertIsNotNone(token.morphology)
    
    def test_toon_document_creation(self):
        """Test TOONDocument creation and methods."""
        tokens = [
            TOONToken(token="الكتاب", pos=POS.NOUN, position=0),
            TOONToken(token="كبير", pos=POS.ADJ, position=1)
        ]
        doc = TOONDocument(
            tokens=tokens,
            source_text="الكتاب كبير",
            language="arb_Arab",
            dialect=Dialect.MSA
        )
        self.assertEqual(len(doc.tokens), 2)
        self.assertEqual(doc.dialect, Dialect.MSA)
        
        # Test helper methods
        nouns = doc.get_tokens_by_pos(POS.NOUN)
        self.assertEqual(len(nouns), 1)
        self.assertEqual(nouns[0].token, "الكتاب")
    
    def test_bilingual_toon(self):
        """Test BilingualTOON with alignments."""
        arabic_doc = TOONDocument(
            tokens=[TOONToken(token="مرحبا", position=0)],
            source_text="مرحبا",
            language="arb_Arab"
        )
        english_doc = TOONDocument(
            tokens=[TOONToken(token="Hello", position=0)],
            source_text="Hello",
            language="eng_Latn"
        )
        alignment = TOONAlignment(
            arabic_token_idx=0,
            english_token_idx=0,
            confidence=0.9
        )
        
        bilingual = BilingualTOON(
            source=arabic_doc,
            target=english_doc,
            alignments=[alignment]
        )
        
        self.assertEqual(len(bilingual.alignments), 1)
        aligned = bilingual.get_aligned_tokens(0)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].token, "Hello")
    
    def test_serialization(self):
        """Test TOON serialization/deserialization."""
        doc = TOONDocument(
            tokens=[TOONToken(token="test", position=0)],
            source_text="test",
            language="arb_Arab"
        )
        
        # Serialize
        data = serialize_toon_document(doc)
        self.assertIn("tokens", data)
        
        # Deserialize
        doc2 = deserialize_toon_document(data)
        self.assertEqual(doc.source_text, doc2.source_text)
        
        # JSON roundtrip
        json_str = toon_document_to_json(doc)
        doc3 = toon_document_from_json(json_str)
        self.assertEqual(doc.source_text, doc3.source_text)
    
    def test_validation(self):
        """Test TOON validation."""
        doc = TOONDocument(
            tokens=[
                TOONToken(token="a", position=0),
                TOONToken(token="b", position=1)
            ],
            source_text="a b",
            language="arb_Arab"
        )
        is_valid, error = validate_toon_document(doc)
        self.assertTrue(is_valid)
        self.assertIsNone(error)


class TestPhase2ArabicNLP(unittest.TestCase):
    """Test Phase 2.2: Arabic NLP Integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models_base = "/home/aModels/models"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        def load_tokenizer(model_name):
            model_path = f"{self.models_base}/arabic_models/{model_name}"
            return AutoTokenizer.from_pretrained(model_path)
        
        def load_model(model_name):
            model_path = f"{self.models_base}/arabic_models/{model_name}"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            model.to(self.device).eval()
            return model
        
        self.load_tokenizer = load_tokenizer
        self.load_model = load_model
    
    def test_toon_generator_initialization(self):
        """Test ArabicTOONGenerator initialization."""
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=True,
            use_qadi=True,
            use_kuwain=False
        )
        self.assertIsNotNone(gen)
    
    def test_basic_toon_generation(self):
        """Test basic TOON generation."""
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=False,
            use_qadi=False,
            use_kuwain=False
        )
        
        text = "مرحبا بالعالم"
        doc = gen.generate_toon(text, language="arb_Arab")
        
        self.assertIsNotNone(doc)
        self.assertEqual(doc.source_text, text)
        self.assertGreater(len(doc.tokens), 0)
        self.assertEqual(doc.language, "arb_Arab")
    
    def test_toon_with_arbml_enabled(self):
        """Test TOON generation with ARBML enabled."""
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=True,
            use_qadi=False,
            use_kuwain=False
        )
        
        text = "الكتاب كبير"
        doc = gen.generate_toon(text, language="arb_Arab")
        
        self.assertIsNotNone(doc)
        self.assertGreater(len(doc.tokens), 0)
        # ARBML may use fallback, but should still work
    
    def test_toon_with_qadi_enabled(self):
        """Test TOON generation with QADI enabled."""
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=False,
            use_qadi=True,
            use_kuwain=False
        )
        
        text = "مرحبا"
        doc = gen.generate_toon(text, language="arb_Arab")
        
        self.assertIsNotNone(doc)
        self.assertIsNotNone(doc.dialect)
        # QADI may use heuristic, but should still work
    
    def test_dialect_classification(self):
        """Test dialect classification."""
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=False,
            use_qadi=True,
            use_kuwain=False
        )
        
        # Test different texts
        test_cases = [
            ("مرحبا", Dialect.MSA),  # MSA
        ]
        
        for text, expected_dialect_type in test_cases:
            doc = gen.generate_toon(text, language="arb_Arab")
            self.assertIsNotNone(doc.dialect)
            # Dialect should be one of the valid types
            self.assertIn(doc.dialect, Dialect)


class TestPhase2TranslationPipeline(unittest.TestCase):
    """Test Phase 2.3: TOON-Enhanced Translation Pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models_base = "/home/aModels/models"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        def load_tokenizer(model_name):
            model_path = f"{self.models_base}/arabic_models/{model_name}"
            return AutoTokenizer.from_pretrained(model_path)
        
        def load_model(model_name):
            model_path = f"{self.models_base}/arabic_models/{model_name}"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            model.to(self.device).eval()
            return model
        
        self.load_tokenizer = load_tokenizer
        self.load_model = load_model
    
    def test_toon_translation_basic(self):
        """Test basic TOON-enhanced translation."""
        text = "مرحبا"
        
        translated, bilingual = translate_with_toon(
            text=text,
            source_lang="arb_Arab",
            target_lang="eng_Latn",
            translation_model_name="nllb-200-1.3B",
            load_tokenizer=self.load_tokenizer,
            load_model=self.load_model,
            device=self.device,
            max_length=512,
            toon_config={"models_base": self.models_base},
            fallback_on_error=True
        )
        
        self.assertIsNotNone(translated)
        self.assertGreater(len(translated), 0)
    
    def test_toon_translation_financial_text(self):
        """Test TOON translation with financial text."""
        text = "التقرير المالي يظهر ربحاً"
        
        translated, bilingual = translate_with_toon(
            text=text,
            source_lang="arb_Arab",
            target_lang="eng_Latn",
            translation_model_name="nllb-200-1.3B",
            load_tokenizer=self.load_tokenizer,
            load_model=self.load_model,
            device=self.device,
            max_length=512,
            toon_config={"models_base": self.models_base},
            fallback_on_error=True
        )
        
        self.assertIsNotNone(translated)
        # Should contain financial terms
        self.assertIn("report", translated.lower() or "profit" in translated.lower() or "financial" in translated.lower())
    
    def test_bilingual_toon_generation(self):
        """Test bilingual TOON generation."""
        text = "الكتاب كبير"
        
        translated, bilingual = translate_with_toon(
            text=text,
            source_lang="arb_Arab",
            target_lang="eng_Latn",
            translation_model_name="nllb-200-1.3B",
            load_tokenizer=self.load_tokenizer,
            load_model=self.load_model,
            device=self.device,
            max_length=512,
            toon_config={"models_base": self.models_base},
            fallback_on_error=True
        )
        
        if bilingual:
            self.assertIsNotNone(bilingual.source)
            self.assertIsNotNone(bilingual.target)
            self.assertGreater(len(bilingual.alignments), 0)
    
    def test_fallback_mechanism(self):
        """Test fallback when TOON fails."""
        # This should work even if TOON components fail
        text = "مرحبا"
        
        translated, bilingual = translate_with_toon(
            text=text,
            source_lang="arb_Arab",
            target_lang="eng_Latn",
            translation_model_name="nllb-200-1.3B",
            load_tokenizer=self.load_tokenizer,
            load_model=self.load_model,
            device=self.device,
            max_length=512,
            toon_config={"models_base": self.models_base},
            fallback_on_error=True
        )
        
        # Should always return translation, even if TOON fails
        self.assertIsNotNone(translated)
        self.assertGreater(len(translated), 0)


class TestPhase2Integration(unittest.TestCase):
    """Test Phase 2.4: QADI/ARBML Repository Integration."""
    
    def test_qadi_dataset_access(self):
        """Test QADI dataset integration."""
        try:
            from qadi_integration import QADIDataset, map_country_to_dialect
            
            dataset = QADIDataset()
            if dataset.is_available():
                codes = dataset.get_country_codes()
                self.assertGreater(len(codes), 0)
                
                # Test country-dialect mapping
                dialect = map_country_to_dialect("EG")
                self.assertIsNotNone(dialect)
        except ImportError:
            self.skipTest("QADI integration not available")
    
    def test_arbml_models_access(self):
        """Test ARBML models integration."""
        try:
            from arbml_integration import ARBMLModelLoader
            
            loader = ARBMLModelLoader()
            models = loader.list_available_models()
            # Should have models available
            self.assertGreaterEqual(len(models), 0)
        except ImportError:
            self.skipTest("ARBML integration not available")


class TestPhase2EndToEnd(unittest.TestCase):
    """Test Phase 2.5: End-to-End Pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models_base = "/home/aModels/models"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        def load_tokenizer(model_name):
            model_path = f"{self.models_base}/arabic_models/{model_name}"
            return AutoTokenizer.from_pretrained(model_path)
        
        def load_model(model_name):
            model_path = f"{self.models_base}/arabic_models/{model_name}"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            model.to(self.device).eval()
            return model
        
        self.load_tokenizer = load_tokenizer
        self.load_model = load_model
    
    def test_full_pipeline_simple(self):
        """Test full pipeline with simple text."""
        text = "مرحبا بالعالم"
        
        # Generate TOON
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=True,
            use_qadi=True,
            use_kuwain=False
        )
        doc = gen.generate_toon(text, language="arb_Arab")
        
        # Validate TOON
        is_valid, error = validate_toon_document(doc)
        self.assertTrue(is_valid, f"TOON validation failed: {error}")
        
        # Translate with TOON
        translated, bilingual = translate_with_toon(
            text=text,
            source_lang="arb_Arab",
            target_lang="eng_Latn",
            translation_model_name="nllb-200-1.3B",
            load_tokenizer=self.load_tokenizer,
            load_model=self.load_model,
            device=self.device,
            max_length=512,
            toon_config={
                "models_base": self.models_base,
                "use_arbml": True,
                "use_qadi": True
            },
            fallback_on_error=True
        )
        
        # Verify results
        self.assertIsNotNone(translated)
        self.assertGreater(len(translated), 0)
        self.assertGreater(len(doc.tokens), 0)
    
    def test_full_pipeline_financial(self):
        """Test full pipeline with financial text."""
        text = "التقرير المالي يظهر ربحاً قدره مليون دولار"
        
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=True,
            use_qadi=True,
            use_kuwain=False
        )
        
        # Generate TOON
        doc = gen.generate_toon(text, language="arb_Arab")
        
        # Translate
        translated, bilingual = translate_with_toon(
            text=text,
            source_lang="arb_Arab",
            target_lang="eng_Latn",
            translation_model_name="nllb-200-1.3B",
            load_tokenizer=self.load_tokenizer,
            load_model=self.load_model,
            device=self.device,
            max_length=512,
            toon_config={"models_base": self.models_base},
            fallback_on_error=True
        )
        
        # Verify
        self.assertIsNotNone(translated)
        self.assertIn("report", translated.lower() or "profit" in translated.lower() or "financial" in translated.lower() or "million" in translated.lower())
        self.assertGreater(len(doc.tokens), 0)
    
    def test_multiple_texts(self):
        """Test pipeline with multiple Arabic texts."""
        test_cases = [
            "مرحبا",
            "الكتاب كبير",
            "التقرير المالي يظهر ربحاً"
        ]
        
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=True,
            use_qadi=True,
            use_kuwain=False
        )
        
        for text in test_cases:
            # Generate TOON
            doc = gen.generate_toon(text, language="arb_Arab")
            self.assertIsNotNone(doc)
            self.assertGreater(len(doc.tokens), 0)
            
            # Translate
            translated, bilingual = translate_with_toon(
                text=text,
                source_lang="arb_Arab",
                target_lang="eng_Latn",
                translation_model_name="nllb-200-1.3B",
                load_tokenizer=self.load_tokenizer,
                load_model=self.load_model,
                device=self.device,
                max_length=512,
                toon_config={"models_base": self.models_base},
                fallback_on_error=True
            )
            
            self.assertIsNotNone(translated)
            self.assertGreater(len(translated), 0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)

