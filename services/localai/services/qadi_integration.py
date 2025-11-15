"""Integration module for QADI dataset and dialect identification.

QADI (https://github.com/qcri/QADI) is a dataset for Arabic dialect identification.
This module provides utilities to work with QADI data alongside camel-tools.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# QADI repository path
QADI_BASE = Path(__file__).parent.parent / "third_party" / "QADI"
QADI_DATASET = QADI_BASE / "dataset" if QADI_BASE.exists() else None
QADI_TESTSET = QADI_BASE / "testset" if QADI_BASE.exists() else None

# Country code to dialect mapping
COUNTRY_TO_DIALECT: Dict[str, str] = {
    "AE": "GULF",      # United Arab Emirates
    "BH": "GULF",      # Bahrain
    "DZ": "MAGHREBI",  # Algeria
    "EG": "EGYPTIAN",  # Egypt
    "IQ": "MESOPOTAMIAN",  # Iraq (similar to Gulf)
    "JO": "LEVANTINE", # Jordan
    "KW": "GULF",      # Kuwait
    "LB": "LEVANTINE", # Lebanon
    "LY": "MAGHREBI",  # Libya
    "MA": "MAGHREBI",  # Morocco
    "OM": "GULF",      # Oman
    "PL": "MAGHREBI", # Palestine (similar to Levantine)
    "QA": "GULF",      # Qatar
    "SA": "GULF",      # Saudi Arabia
    "SD": "SUDANESE",  # Sudan
    "SY": "LEVANTINE", # Syria
    "TN": "MAGHREBI",  # Tunisia
    "YE": "YEMENI",    # Yemen
}


class QADIDataset:
    """Wrapper for QADI dataset access."""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        """Initialize QADI dataset access.
        
        Args:
            dataset_path: Path to QADI dataset directory
        """
        self.dataset_path = dataset_path or QADI_DATASET
        self.testset_path = QADI_TESTSET
    
    def get_country_codes(self) -> List[str]:
        """Get list of country codes in QADI dataset.
        
        Returns:
            List of ISO country codes
        """
        if not self.dataset_path or not self.dataset_path.exists():
            return []
        
        country_codes = []
        for file in self.dataset_path.glob("QADI_train_ids_*.txt"):
            # Extract country code from filename: QADI_train_ids_XX.txt
            code = file.stem.split("_")[-1]
            country_codes.append(code)
        
        return sorted(country_codes)
    
    def get_dialect_for_country(self, country_code: str) -> Optional[str]:
        """Get dialect classification for a country code.
        
        Args:
            country_code: ISO country code (e.g., "EG", "SA")
            
        Returns:
            Dialect name or None
        """
        return COUNTRY_TO_DIALECT.get(country_code.upper())
    
    def get_tweet_count(self, country_code: str) -> int:
        """Get number of training tweets for a country.
        
        Args:
            country_code: ISO country code
            
        Returns:
            Number of tweets (0 if not found)
        """
        if not self.dataset_path:
            return 0
        
        file_path = self.dataset_path / f"QADI_train_ids_{country_code.upper()}.txt"
        if not file_path.exists():
            return 0
        
        try:
            with open(file_path, 'r') as f:
                return len([line for line in f if line.strip()])
        except Exception as e:
            logger.error(f"Failed to count tweets for {country_code}: {e}")
            return 0
    
    def is_available(self) -> bool:
        """Check if QADI dataset is available.
        
        Returns:
            True if dataset directory exists
        """
        return self.dataset_path is not None and self.dataset_path.exists()


def map_country_to_dialect(country_code: str) -> Optional[str]:
    """Map QADI country code to dialect.
    
    Args:
        country_code: ISO country code from QADI
        
    Returns:
        Dialect name (MSA, EGYPTIAN, LEVANTINE, GULF, MAGHREBI, etc.)
    """
    return COUNTRY_TO_DIALECT.get(country_code.upper())


def get_qadi_statistics() -> Dict[str, Any]:
    """Get statistics about QADI dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    dataset = QADIDataset()
    
    if not dataset.is_available():
        return {"available": False}
    
    stats = {
        "available": True,
        "countries": len(dataset.get_country_codes()),
        "country_codes": dataset.get_country_codes(),
        "total_tweets": 0
    }
    
    for code in dataset.get_country_codes():
        count = dataset.get_tweet_count(code)
        stats["total_tweets"] += count
        stats[f"{code}_tweets"] = count
        stats[f"{code}_dialect"] = dataset.get_dialect_for_country(code)
    
    return stats

