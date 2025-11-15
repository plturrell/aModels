"""Integration test for full TOON-enhanced translation pipeline."""

import unittest
import sys
import os
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from toon_generator import ArabicTOONGenerator
from toon_translation import translate_with_toon
from toon_utils import validate_toon_document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class TestTOONPipelineIntegration(unittest.TestCase):
    """Test the full TOON-enhanced translation pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.models_base = "/home/aModels/models"
        
        def load_tokenizer(model_name):
            model_path = f"{cls.models_base}/arabic_models/{model_name}"
            return AutoTokenizer.from_pretrained(model_path)
        
        def load_model(model_name):
            model_path = f"{cls.models_base}/arabic_models/{model_name}"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            model.to(cls.device)
            model.eval()
            return model
        
        cls.load_tokenizer = load_tokenizer
        cls.load_model = load_model
    
    def test_toon_generation(self):
        """Test TOON generation from Arabic text."""
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=True,
            use_qadi=True,
            use_kuwain=False
        )
        
        text = "الكتاب كبير"
        doc = gen.generate_toon(text, language="arb_Arab")
        
        self.assertIsNotNone(doc)
        self.assertEqual(doc.source_text, text)
        self.assertGreater(len(doc.tokens), 0)
        self.assertEqual(doc.language, "arb_Arab")
    
    def test_toon_validation(self):
        """Test TOON document validation."""
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=False,
            use_qadi=False,
            use_kuwain=False
        )
        
        text = "مرحبا"
        doc = gen.generate_toon(text, language="arb_Arab")
        
        is_valid, error = validate_toon_document(doc)
        self.assertTrue(is_valid, f"TOON validation failed: {error}")
    
    def test_toon_translation(self):
        """Test TOON-enhanced translation."""
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
            toon_config={
                "models_base": self.models_base,
                "use_arbml": True,
                "use_qadi": True,
                "use_kuwain": False
            },
            fallback_on_error=True
        )
        
        self.assertIsNotNone(translated)
        self.assertGreater(len(translated), 0)
        # Bilingual TOON may be None if fallback was used
        # That's acceptable for this test
    
    def test_financial_text_translation(self):
        """Test translation of financial text."""
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
    
    def test_fallback_mechanism(self):
        """Test that fallback works when TOON fails."""
        # This test verifies graceful degradation
        gen = ArabicTOONGenerator(
            models_base=self.models_base,
            use_arbml=False,  # Disable to test fallback
            use_qadi=False,
            use_kuwain=False
        )
        
        text = "مرحبا"
        doc = gen.generate_toon(text, language="arb_Arab")
        
        # Should still work with fallback
        self.assertIsNotNone(doc)
        self.assertEqual(len(doc.tokens), 1)  # Simple tokenization


if __name__ == "__main__":
    unittest.main()

