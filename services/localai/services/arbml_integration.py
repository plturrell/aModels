"""Integration module for ARBML repository models and tools.

ARBML (https://github.com/ARBML/ARBML) provides pre-trained Arabic ML models.
This module provides wrappers to use ARBML models alongside camel-tools.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ARBML repository path
ARBML_BASE = Path(__file__).parent.parent / "third_party" / "ARBML"
ARBML_MODELS = ARBML_BASE / "models" / "Keras" if ARBML_BASE.exists() else None

# Try to import TensorFlow/Keras for ARBML models
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available, ARBML models cannot be loaded")


class ARBMLModelLoader:
    """Loader for ARBML pre-trained models."""
    
    def __init__(self, models_path: Optional[Path] = None):
        """Initialize ARBML model loader.
        
        Args:
            models_path: Path to ARBML models directory
        """
        self.models_path = models_path or ARBML_MODELS
        self.loaded_models: Dict[str, Any] = {}
    
    def load_diacritization_model(self) -> Optional[Any]:
        """Load ARBML Arabic diacritization model.
        
        Returns:
            Keras model or None if not available
        """
        if not HAS_TENSORFLOW:
            logger.warning("TensorFlow not available for ARBML diacritization")
            return None
        
        if not self.models_path:
            logger.warning("ARBML models path not found")
            return None
        
        model_file = self.models_path / "diactrization.h5"
        if not model_file.exists():
            logger.warning(f"ARBML diacritization model not found: {model_file}")
            return None
        
        try:
            model = tf.keras.models.load_model(str(model_file))
            self.loaded_models["diacritization"] = model
            logger.info("✅ ARBML diacritization model loaded")
            return model
        except Exception as e:
            logger.error(f"Failed to load ARBML diacritization model: {e}")
            return None
    
    def load_sentiment_model(self) -> Optional[Any]:
        """Load ARBML Arabic sentiment classification model.
        
        Returns:
            Keras model or None if not available
        """
        if not HAS_TENSORFLOW:
            return None
        
        if not self.models_path:
            return None
        
        model_file = self.models_path / "sentiment_classification.h5"
        if not model_file.exists():
            logger.debug("ARBML sentiment model not found")
            return None
        
        try:
            model = tf.keras.models.load_model(str(model_file))
            self.loaded_models["sentiment"] = model
            logger.info("✅ ARBML sentiment model loaded")
            return model
        except Exception as e:
            logger.error(f"Failed to load ARBML sentiment model: {e}")
            return None
    
    def list_available_models(self) -> list[str]:
        """List available ARBML models.
        
        Returns:
            List of available model names
        """
        if not self.models_path or not self.models_path.exists():
            return []
        
        models = []
        for model_file in self.models_path.glob("*.h5"):
            models.append(model_file.stem)
        
        return models


def add_diacritization(text: str, model: Optional[Any] = None) -> str:
    """Add diacritics to Arabic text using ARBML model.
    
    Args:
        text: Arabic text without diacritics
        model: ARBML diacritization model (will load if None)
        
    Returns:
        Text with diacritics
    """
    if model is None:
        loader = ARBMLModelLoader()
        model = loader.load_diacritization_model()
    
    if model is None:
        logger.warning("ARBML diacritization model not available")
        return text
    
    try:
        # This is a placeholder - actual implementation depends on model input format
        # ARBML models typically require specific preprocessing
        logger.debug("ARBML diacritization would be applied here")
        return text  # Placeholder
    except Exception as e:
        logger.error(f"Diacritization failed: {e}")
        return text

