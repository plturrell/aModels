# QADI and ARBML Repository Integration

## Overview

This document describes the integration of QADI and ARBML repositories into the TOON translation pipeline.

## Repositories

### QADI (QCRI Arabic Dialect Identification)
- **Repository**: https://github.com/qcri/QADI.git
- **Type**: Dataset repository
- **Content**: 540,590 tweets from 18 Arab countries
- **Purpose**: Benchmarking and training data for dialect identification
- **Location**: `/home/aModels/services/localai/third_party/QADI/`

### ARBML (Arabic Machine Learning)
- **Repository**: https://github.com/ARBML/ARBML.git
- **Type**: Collection of Arabic ML models and demos
- **Content**: Pre-trained Keras models, web interfaces, notebooks
- **Models**: Diacritization, Translation, Sentiment, Word Embeddings, etc.
- **Location**: `/home/aModels/services/localai/third_party/ARBML/`

## Integration Status

### Current Implementation

**Primary Tools (camel-tools)**:
- ✅ ARBML morphological analysis via `camel_tools.morphology`
- ✅ QADI dialect identification via `camel_tools.dialectid`
- ✅ These are the main tools used in production

**Supplementary Repositories**:
- ✅ QADI dataset: Available for reference and evaluation
- ✅ ARBML models: Available for additional features (diacritization, etc.)

## Integration Modules

### `qadi_integration.py`
Provides utilities to work with QADI dataset:
- Country code to dialect mapping
- Dataset statistics
- Tweet count per country
- Reference data for dialect classification

**Usage**:
```python
from qadi_integration import QADIDataset, map_country_to_dialect

dataset = QADIDataset()
dialect = map_country_to_dialect("EG")  # Returns "EGYPTIAN"
```

### `arbml_integration.py`
Provides wrappers for ARBML pre-trained models:
- Arabic diacritization model
- Sentiment classification model
- Model loader utilities

**Usage**:
```python
from arbml_integration import ARBMLModelLoader

loader = ARBMLModelLoader()
diacritization_model = loader.load_diacritization_model()
```

## QADI Dataset Structure

```
QADI/
├── dataset/
│   ├── QADI_train_ids_AE.txt  (UAE)
│   ├── QADI_train_ids_EG.txt  (Egypt)
│   ├── QADI_train_ids_SA.txt  (Saudi Arabia)
│   └── ... (18 countries)
└── testset/
    └── QADI_test.txt
```

**Country Codes**:
- AE, BH, KW, QA, SA, OM → GULF dialect
- EG → EGYPTIAN dialect
- JO, LB, SY, PL → LEVANTINE dialect
- DZ, LY, MA, TN → MAGHREBI dialect
- SD → SUDANESE dialect
- YE → YEMENI dialect

## ARBML Models Available

Located in `/third_party/ARBML/models/Keras/`:
- `diactrization.h5` - Arabic diacritization
- `sentiment_classification.h5` - Sentiment analysis
- `digits_classification.h5` - Digit recognition
- `letters_classification.h5` - Letter recognition
- `poem_generation.h5` - Poem generation
- `Translation/` - Translation models

## Integration with TOON Generator

The TOON generator uses:
1. **Primary**: camel-tools for ARBML/QADI functionality
2. **Supplementary**: QADI dataset for country-dialect mapping
3. **Optional**: ARBML models for additional features

**Code Integration**:
```python
from toon_generator import ArabicTOONGenerator
from qadi_integration import map_country_to_dialect

# TOON generator uses camel-tools (primary)
gen = ArabicTOONGenerator(use_arbml=True, use_qadi=True)

# QADI dataset provides reference mapping
dialect = map_country_to_dialect("EG")  # "EGYPTIAN"
```

## Benefits

1. **QADI Dataset**:
   - Provides country-level dialect ground truth
   - Useful for evaluation and benchmarking
   - Reference for dialect classification

2. **ARBML Models**:
   - Additional Arabic NLP capabilities
   - Diacritization support
   - Sentiment analysis
   - Can enhance TOON with diacritics

## Usage Notes

- **camel-tools remains primary**: The main ARBML/QADI functionality comes from camel-tools
- **Repositories are supplementary**: QADI/ARBML repos provide datasets and additional models
- **Optional features**: ARBML models require TensorFlow/Keras
- **Dataset access**: QADI dataset is read-only (tweet IDs, not full tweets)

## Future Enhancements

1. Use ARBML diacritization model to add diacritics to TOON tokens
2. Use QADI dataset for evaluation metrics
3. Train custom models using QADI data
4. Integrate ARBML sentiment analysis into TOON pipeline

## Files

- `services/qadi_integration.py` - QADI dataset utilities
- `services/arbml_integration.py` - ARBML model loaders
- `third_party/QADI/` - QADI repository
- `third_party/ARBML/` - ARBML repository

