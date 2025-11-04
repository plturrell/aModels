"""Training service for aModels.

This package provides training pipeline integration with Glean Catalog,
pattern learning, and model training orchestration.
"""

from .glean_integration import (
    GleanTrainingClient,
    ingest_glean_data_for_training,
)
from .extract_client import ExtractServiceClient
from .pipeline import TrainingPipeline
from .pattern_learning import (
    PatternLearningEngine,
    ColumnTypePatternLearner,
    RelationshipPatternLearner,
    MetadataEntropyPatternLearner,
)
from .evaluation import (
    evaluate_training_results,
    export_training_metrics_to_glean,
)
from .temporal_analysis import (
    TemporalPatternLearner,
    SchemaEvolutionAnalyzer,
)
from .intelligence_metrics import (
    DomainIntelligenceEvaluator,
)

__all__ = [
    "GleanTrainingClient",
    "ingest_glean_data_for_training",
    "ExtractServiceClient",
    "TrainingPipeline",
    "PatternLearningEngine",
    "ColumnTypePatternLearner",
    "RelationshipPatternLearner",
    "MetadataEntropyPatternLearner",
    "evaluate_training_results",
    "export_training_metrics_to_glean",
    "TemporalPatternLearner",
    "SchemaEvolutionAnalyzer",
    "DomainIntelligenceEvaluator",
]

