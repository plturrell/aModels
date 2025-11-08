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
    WorkflowPatternLearner,
    ColumnTypePatternLearner,
    RelationshipPatternLearner,
    MetadataEntropyPatternLearner,
)

# Deep learning pattern learners (Phase 7.1)
try:
    from .pattern_learning_gnn import GNNRelationshipPatternLearner
    from .sequence_pattern_transformer import SequencePatternTransformer
    HAS_DEEP_LEARNING = True
except ImportError:
    HAS_DEEP_LEARNING = False
    GNNRelationshipPatternLearner = None
    SequencePatternTransformer = None

# GNN modules (5 core implementations)
try:
    from .gnn_node_classifier import GNNNodeClassifier
    from .gnn_link_predictor import GNNLinkPredictor
    from .gnn_embeddings import GNNEmbedder
    from .gnn_anomaly_detector import GNNAnomalyDetector
    from .gnn_schema_matcher import GNNSchemaMatcher
    HAS_GNN_MODULES = True
except ImportError:
    HAS_GNN_MODULES = False
    GNNNodeClassifier = None
    GNNLinkPredictor = None
    GNNEmbedder = None
    GNNAnomalyDetector = None
    GNNSchemaMatcher = None

# Meta-pattern learner (Phase 7.2)
try:
    from .meta_pattern_learner import MetaPatternLearner
    HAS_META_PATTERNS = True
except ImportError:
    HAS_META_PATTERNS = False
    MetaPatternLearner = None

# Active pattern learner (Phase 7.4)
try:
    from .active_pattern_learner import ActivePatternLearner
    HAS_ACTIVE_LEARNING = True
except ImportError:
    HAS_ACTIVE_LEARNING = False
    ActivePatternLearner = None

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
    "WorkflowPatternLearner",
    "SemanticPatternLearner",
    "SemanticFeatureExtractor",
    "ColumnTypePatternLearner",
    "RelationshipPatternLearner",
    "MetadataEntropyPatternLearner",
    "evaluate_training_results",
    "export_training_metrics_to_glean",
    "TemporalPatternLearner",
    "SchemaEvolutionAnalyzer",
    "DomainIntelligenceEvaluator",
]

# Add deep learning learners if available (Phase 7.1)
if HAS_DEEP_LEARNING:
    if GNNRelationshipPatternLearner is not None:
        __all__.append("GNNRelationshipPatternLearner")
    if SequencePatternTransformer is not None:
        __all__.append("SequencePatternTransformer")

# Add meta-pattern learner if available (Phase 7.2)
if HAS_META_PATTERNS:
    if MetaPatternLearner is not None:
        __all__.append("MetaPatternLearner")

# Add active pattern learner if available (Phase 7.4)
if HAS_ACTIVE_LEARNING:
    if ActivePatternLearner is not None:
        __all__.append("ActivePatternLearner")

# Pattern transfer learner (Phase 8.4)
try:
    from .pattern_transfer import PatternTransferLearner
    HAS_PATTERN_TRANSFER = True
except ImportError:
    HAS_PATTERN_TRANSFER = False
    PatternTransferLearner = None

# Add pattern transfer learner if available (Phase 8.4)
if HAS_PATTERN_TRANSFER:
    if PatternTransferLearner is not None:
        __all__.append("PatternTransferLearner")

# Auto-tuner (Phase 9.1)
try:
    from .auto_tuner import AutoTuner
    HAS_AUTO_TUNER = True
except ImportError:
    HAS_AUTO_TUNER = False
    AutoTuner = None

# Add auto-tuner if available (Phase 9.1)
if HAS_AUTO_TUNER:
    if AutoTuner is not None:
        __all__.append("AutoTuner")

# Add GNN modules if available
if HAS_GNN_MODULES:
    if GNNNodeClassifier is not None:
        __all__.append("GNNNodeClassifier")
    if GNNLinkPredictor is not None:
        __all__.append("GNNLinkPredictor")
    if GNNEmbedder is not None:
        __all__.append("GNNEmbedder")
    if GNNAnomalyDetector is not None:
        __all__.append("GNNAnomalyDetector")
    if GNNSchemaMatcher is not None:
        __all__.append("GNNSchemaMatcher")

