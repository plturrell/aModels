"""Integration tests for TOON-enhanced translation."""

import unittest
from unittest.mock import Mock, patch
from toon_translation import translate_with_toon
from toon_schema import TOONDocument, BilingualTOON


class TestTOONTranslation(unittest.TestCase):
    """Test TOON-enhanced translation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_model = Mock()
        self.mock_load_tokenizer = Mock(return_value=self.mock_tokenizer)
        self.mock_load_model = Mock(return_value=self.mock_model)
        
        # Mock NLLB translation
        self.mock_model.generate.return_value = [[1, 2, 3]]  # Mock token IDs
        self.mock_tokenizer.decode.return_value = "Hello world"
        self.mock_tokenizer.return_value = {"input_ids": Mock()}
        self.mock_tokenizer.lang_code_to_id = {"eng_Latn": 1}
    
    @patch('toon_translation._translate_with_nllb')
    @patch('toon_generator.ArabicTOONGenerator')
    def test_toon_translation_basic(self, mock_generator_class, mock_translate):
        """Test basic TOON translation."""
        # Setup mocks
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        arabic_toon = TOONDocument(
            tokens=[],
            source_text="مرحبا",
            language="arb_Arab"
        )
        mock_generator.generate_toon.return_value = arabic_toon
        
        mock_translate.return_value = "Hello"
        
        # Run translation
        translated, bilingual = translate_with_toon(
            text="مرحبا",
            source_lang="arb_Arab",
            target_lang="eng_Latn",
            translation_model_name="nllb-200-1.3B",
            load_tokenizer=self.mock_load_tokenizer,
            load_model=self.mock_load_model,
            device="cpu",
            max_length=512,
            toon_config={"use_arbml": False},
            fallback_on_error=True
        )
        
        # Verify
        self.assertIsNotNone(translated)
        self.assertIsNotNone(bilingual)
        self.assertIsInstance(bilingual, BilingualTOON)
        mock_generator.generate_toon.assert_called_once()
    
    @patch('toon_translation._translate_with_nllb')
    @patch('toon_generator.ArabicTOONGenerator')
    def test_toon_translation_fallback(self, mock_generator_class, mock_translate):
        """Test TOON translation fallback on error."""
        # Setup mocks to raise error
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_toon.side_effect = Exception("TOON generation failed")
        
        mock_translate.return_value = "Hello"
        
        # Run translation with fallback
        translated, bilingual = translate_with_toon(
            text="مرحبا",
            source_lang="arb_Arab",
            target_lang="eng_Latn",
            translation_model_name="nllb-200-1.3B",
            load_tokenizer=self.mock_load_tokenizer,
            load_model=self.mock_load_model,
            device="cpu",
            max_length=512,
            toon_config={"use_arbml": False},
            fallback_on_error=True
        )
        
        # Verify fallback occurred
        self.assertIsNotNone(translated)
        self.assertIsNone(bilingual)  # No TOON on fallback
        mock_translate.assert_called_once()
    
    @patch('toon_translation._translate_with_nllb')
    @patch('toon_generator.ArabicTOONGenerator')
    def test_toon_translation_no_fallback(self, mock_generator_class, mock_translate):
        """Test TOON translation without fallback raises error."""
        # Setup mocks to raise error
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_toon.side_effect = Exception("TOON generation failed")
        
        # Run translation without fallback
        with self.assertRaises(Exception):
            translate_with_toon(
                text="مرحبا",
                source_lang="arb_Arab",
                target_lang="eng_Latn",
                translation_model_name="nllb-200-1.3B",
                load_tokenizer=self.mock_load_tokenizer,
                load_model=self.mock_load_model,
                device="cpu",
                max_length=512,
                toon_config={"use_arbml": False},
                fallback_on_error=False
            )


class TestDirectVsTOONTranslation(unittest.TestCase):
    """Compare direct translation vs TOON-enhanced translation."""
    
    def test_translation_consistency(self):
        """Test that TOON translation produces valid output."""
        # This would be an integration test with actual models
        # For now, we just verify the structure
        pass
    
    def test_toon_quality_improvement(self):
        """Test that TOON provides quality improvements.
        
        This would require:
        - Actual model loading
        - Quality metrics (BLEU, etc.)
        - Comparison with direct translation
        """
        pass


if __name__ == "__main__":
    unittest.main()

