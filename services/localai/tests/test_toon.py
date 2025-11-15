"""Unit tests for TOON schema and generation."""

import unittest
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
    validate_toon_document,
    serialize_toon_document,
    deserialize_toon_document,
    toon_document_to_json,
    toon_document_from_json
)
from toon_generator import ArabicTOONGenerator


class TestTOONSchema(unittest.TestCase):
    """Test TOON schema definitions."""
    
    def test_toon_token_creation(self):
        """Test creating a TOONToken."""
        token = TOONToken(
            token="الكلمة",
            lemma="كلم",
            pos=POS.NOUN,
            position=0
        )
        self.assertEqual(token.token, "الكلمة")
        self.assertEqual(token.lemma, "كلم")
        self.assertEqual(token.pos, POS.NOUN)
        self.assertEqual(token.position, 0)
    
    def test_toon_morphology(self):
        """Test TOONMorphology."""
        morph = TOONMorphology(
            gender="feminine",
            number="singular",
            case="nominative"
        )
        self.assertEqual(morph.gender, "feminine")
        self.assertEqual(morph.number, "singular")
        self.assertEqual(morph.case, "nominative")
    
    def test_toon_document_creation(self):
        """Test creating a TOONDocument."""
        tokens = [
            TOONToken(token="مرحبا", position=0),
            TOONToken(token="بالعالم", position=1)
        ]
        doc = TOONDocument(
            tokens=tokens,
            source_text="مرحبا بالعالم",
            language="arb_Arab"
        )
        self.assertEqual(len(doc.tokens), 2)
        self.assertEqual(doc.source_text, "مرحبا بالعالم")
        self.assertEqual(doc.language, "arb_Arab")
    
    def test_toon_document_methods(self):
        """Test TOONDocument helper methods."""
        tokens = [
            TOONToken(token="الكتاب", pos=POS.NOUN, position=0),
            TOONToken(token="كبير", pos=POS.ADJ, position=1),
            TOONToken(token="الكتاب", pos=POS.NOUN, position=2)
        ]
        doc = TOONDocument(
            tokens=tokens,
            source_text="الكتاب كبير الكتاب",
            language="arb_Arab"
        )
        
        nouns = doc.get_tokens_by_pos(POS.NOUN)
        self.assertEqual(len(nouns), 2)
        
        text = doc.to_text()
        self.assertEqual(text, "الكتاب كبير الكتاب")
    
    def test_toon_alignment(self):
        """Test TOONAlignment."""
        alignment = TOONAlignment(
            arabic_token_idx=0,
            english_token_idx=0,
            alignment_type="word",
            confidence=0.9
        )
        self.assertEqual(alignment.arabic_token_idx, 0)
        self.assertEqual(alignment.english_token_idx, 0)
        self.assertEqual(alignment.confidence, 0.9)
    
    def test_bilingual_toon(self):
        """Test BilingualTOON."""
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
        
        self.assertEqual(len(bilingual.source.tokens), 1)
        self.assertEqual(len(bilingual.target.tokens), 1)
        self.assertEqual(len(bilingual.alignments), 1)
        
        aligned = bilingual.get_aligned_tokens(0)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].token, "Hello")


class TestTOONUtils(unittest.TestCase):
    """Test TOON utilities."""
    
    def test_serialization(self):
        """Test TOON serialization."""
        doc = TOONDocument(
            tokens=[TOONToken(token="test", position=0)],
            source_text="test",
            language="arb_Arab"
        )
        
        # Serialize
        data = serialize_toon_document(doc)
        self.assertIn("tokens", data)
        self.assertIn("source_text", data)
        
        # Deserialize
        doc2 = deserialize_toon_document(data)
        self.assertEqual(doc.source_text, doc2.source_text)
        self.assertEqual(len(doc.tokens), len(doc2.tokens))
    
    def test_json_serialization(self):
        """Test JSON serialization."""
        doc = TOONDocument(
            tokens=[TOONToken(token="test", position=0)],
            source_text="test",
            language="arb_Arab"
        )
        
        # To JSON
        json_str = toon_document_to_json(doc)
        self.assertIn("test", json_str)
        
        # From JSON
        doc2 = toon_document_from_json(json_str)
        self.assertEqual(doc.source_text, doc2.source_text)
    
    def test_validation(self):
        """Test TOON validation."""
        # Valid document
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
        
        # Invalid document (duplicate positions)
        doc2 = TOONDocument(
            tokens=[
                TOONToken(token="a", position=0),
                TOONToken(token="b", position=0)  # Duplicate
            ],
            source_text="a b",
            language="arb_Arab"
        )
        is_valid2, error2 = validate_toon_document(doc2)
        self.assertFalse(is_valid2)
        self.assertIsNotNone(error2)


class TestTOONGenerator(unittest.TestCase):
    """Test TOON generator."""
    
    def test_generator_initialization(self):
        """Test ArabicTOONGenerator initialization."""
        generator = ArabicTOONGenerator(
            use_arbml=False,
            use_qadi=False,
            use_kuwain=False
        )
        self.assertIsNotNone(generator)
        self.assertFalse(generator.use_arbml)
        self.assertFalse(generator.use_qadi)
        self.assertFalse(generator.use_kuwain)
    
    def test_basic_toon_generation(self):
        """Test basic TOON generation without ARBML."""
        generator = ArabicTOONGenerator(
            use_arbml=False,
            use_qadi=False,
            use_kuwain=False
        )
        
        text = "مرحبا بالعالم"
        doc = generator.generate_toon(text)
        
        self.assertIsNotNone(doc)
        self.assertEqual(doc.source_text, text)
        self.assertGreater(len(doc.tokens), 0)
        self.assertEqual(doc.language, "arb_Arab")
    
    def test_tokenization(self):
        """Test tokenization."""
        generator = ArabicTOONGenerator(use_arbml=False)
        
        text = "مرحبا بالعالم"
        tokens = generator._tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_normalization(self):
        """Test text normalization."""
        generator = ArabicTOONGenerator(use_arbml=False)
        
        text = "مرحبا   بالعالم"
        normalized = generator._normalize_text(text)
        
        # Should remove extra whitespace
        self.assertNotIn("  ", normalized)


if __name__ == "__main__":
    unittest.main()

