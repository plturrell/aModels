#!/usr/bin/env python3
"""
Training Service - FastAPI HTTP Server

Exposes training pipeline, pattern learning, and domain training as HTTP endpoints.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import time
from datetime import timedelta

# Add parent directory to path for imports
# This allows importing from 'training' package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Change to parent directory to allow relative imports
os.chdir(parent_dir)

# Import training modules
from training.pipeline import TrainingPipeline
from training.pattern_learning import PatternLearningEngine
from training.domain_trainer import DomainTrainer
from training.domain_metrics import DomainMetricsCollector
from training.ab_testing import ABTestManager
from training.rollback_manager import RollbackManager
from training.routing_optimizer import RoutingOptimizer
from training.domain_optimizer import DomainOptimizer
from training.domain_filter import DomainFilter, PrivacyConfig

# Try to import optional modules
try:
    from training.pattern_learning_gnn import GNNRelationshipPatternLearner
    HAS_GNN = True
except ImportError:
    HAS_GNN = False

# Import GNN modules (Priority 1-4)
try:
    from training.gnn_embeddings import GNNEmbedder
    from training.gnn_node_classifier import GNNNodeClassifier
    from training.gnn_link_predictor import GNNLinkPredictor
    from training.gnn_anomaly_detector import GNNAnomalyDetector
    from training.gnn_schema_matcher import GNNSchemaMatcher
    HAS_GNN_MODULES = True
except ImportError:
    HAS_GNN_MODULES = False
    GNNEmbedder = None
    GNNNodeClassifier = None
    GNNLinkPredictor = None
    GNNAnomalyDetector = None
    GNNSchemaMatcher = None

# Import GNN API models
try:
    from training.api.gnn_models import (
        GNNEmbeddingsRequest,
        GNNClassifyRequest,
        GNNPredictLinksRequest,
        GNNStructuralInsightsRequest,
        GNNDomainQueryRequest
    )
    HAS_GNN_API_MODELS = True
except ImportError:
    HAS_GNN_API_MODELS = False

try:
    from training.meta_pattern_learner import MetaPatternLearner
    HAS_META_PATTERNS = True
except ImportError:
    HAS_META_PATTERNS = False

try:
    from training.sequence_pattern_transformer import SequencePatternTransformer
    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False

try:
    from training.active_pattern_learner import ActivePatternLearner
    HAS_ACTIVE_LEARNING = True
except ImportError:
    HAS_ACTIVE_LEARNING = False

try:
    from training.pattern_transfer import PatternTransferLearner
    HAS_PATTERN_TRANSFER = True
except ImportError:
    HAS_PATTERN_TRANSFER = False

try:
    from training.auto_tuner import AutoTuner
    HAS_AUTO_TUNER = True
except ImportError:
    HAS_AUTO_TUNER = False

# Priority 6: Domain-aware routing
try:
    from training.gnn_domain_registry import GNNDomainRegistry, DomainModelInfo
    from training.gnn_domain_router import GNNDomainRouter
    HAS_GNN_DOMAIN_ROUTING = True
except ImportError:
    HAS_GNN_DOMAIN_ROUTING = False
    GNNDomainRegistry = None
    DomainModelInfo = None
    GNNDomainRouter = None

# Priority 6: Registry API models
try:
    from training.api.gnn_registry_models import (
        RegisterModelRequest,
        ModelInfoResponse,
        DomainModelInfoResponse,
        ListDomainsResponse,
    )
    HAS_REGISTRY_MODELS = True
except ImportError:
    HAS_REGISTRY_MODELS = False
    RegisterModelRequest = None
    ModelInfoResponse = None
    DomainModelInfoResponse = None
    ListDomainsResponse = None

# Priority 7: GNN Cache Manager
try:
    from training.gnn_cache_manager import GNNCacheManager
    HAS_GNN_CACHE = True
except ImportError:
    HAS_GNN_CACHE = False
    GNNCacheManager = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Training Service",
    description="Training pipeline, pattern learning, and domain training service",
    version="1.0.0"
)

# Initialize components
pipeline = None
pattern_engine = None
domain_trainer = None
metrics_collector = None
ab_test_manager = None
rollback_manager = None
routing_optimizer = None
domain_optimizer = None
domain_filter = None

# GNN components
gnn_embedder = None
gnn_classifier = None
gnn_link_predictor = None
gnn_anomaly_detector = None
gnn_schema_matcher = None

# Priority 6: GNN domain registry and router
gnn_domain_registry = None
gnn_domain_router = None

# Priority 7: GNN cache manager
gnn_cache_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize service components on startup."""
    global pipeline, pattern_engine, domain_trainer, metrics_collector
    global ab_test_manager, rollback_manager, routing_optimizer, domain_optimizer, domain_filter
    global gnn_embedder, gnn_classifier, gnn_link_predictor, gnn_anomaly_detector, gnn_schema_matcher
    global gnn_domain_registry, gnn_domain_router, gnn_cache_manager
    
    logger.info("Initializing Training Service components...")
    
    try:
        # Initialize training pipeline
        pipeline = TrainingPipeline(
            extract_service_url=os.getenv("EXTRACT_SERVICE_URL"),
            output_dir=os.getenv("TRAINING_OUTPUT_DIR", "/workspace/data/training")
        )
        
        # Initialize pattern learning engine
        pattern_engine = PatternLearningEngine()
        
        # Initialize domain trainer
        domain_trainer = DomainTrainer(
            localai_url=os.getenv("LOCALAI_URL", "http://localai-compat:8080"),
            postgres_dsn=os.getenv("POSTGRES_DSN"),
            redis_url=os.getenv("REDIS_URL")
        )
        
        # Initialize metrics collector
        metrics_collector = DomainMetricsCollector(
            localai_url=os.getenv("LOCALAI_URL", "http://localai-compat:8080"),
            postgres_dsn=os.getenv("POSTGRES_DSN")
        )
        
        # Initialize A/B test manager
        ab_test_manager = ABTestManager(
            postgres_dsn=os.getenv("POSTGRES_DSN"),
            redis_url=os.getenv("REDIS_URL")
        )
        
        # Initialize rollback manager
        rollback_manager = RollbackManager(
            postgres_dsn=os.getenv("POSTGRES_DSN"),
            redis_url=os.getenv("REDIS_URL"),
            localai_url=os.getenv("LOCALAI_URL", "http://localai-compat:8080")
        )
        
        # Initialize routing optimizer
        routing_optimizer = RoutingOptimizer(
            postgres_dsn=os.getenv("POSTGRES_DSN"),
            learning_rate=float(os.getenv("ROUTING_LEARNING_RATE", "0.1"))
        )
        
        # Initialize domain optimizer
        domain_optimizer = DomainOptimizer(
            redis_url=os.getenv("REDIS_URL"),
            cache_ttl=int(os.getenv("DOMAIN_CACHE_TTL", "3600"))
        )
        
        # Initialize domain filter
        privacy_config = PrivacyConfig(privacy_level="medium")
        domain_filter = DomainFilter(
            localai_url=os.getenv("LOCALAI_URL", "http://localai-compat:8080"),
            privacy_config=privacy_config
        )
        
        # Priority 6: Initialize GNN domain registry and router
        if HAS_GNN_DOMAIN_ROUTING:
            try:
                registry_dir = os.getenv("GNN_REGISTRY_DIR", "./models/gnn_registry")
                gnn_domain_registry = GNNDomainRegistry(registry_dir=registry_dir)
                gnn_domain_router = GNNDomainRouter(registry=gnn_domain_registry)
                logger.info("✅ GNN domain registry and router initialized")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize GNN domain registry: {e}")
                gnn_domain_registry = None
                gnn_domain_router = None
        
        # Priority 7: Initialize GNN cache manager
        if HAS_GNN_CACHE:
            try:
                redis_url = os.getenv("REDIS_URL")
                default_ttl = int(os.getenv("GNN_CACHE_TTL", "3600"))
                cache_dir = os.getenv("GNN_CACHE_DIR", "./gnn_cache")
                max_memory_size = int(os.getenv("GNN_CACHE_MAX_SIZE", "1000"))
                
                gnn_cache_manager = GNNCacheManager(
                    redis_url=redis_url,
                    default_ttl=default_ttl,
                    cache_dir=cache_dir,
                    max_memory_size=max_memory_size,
                    enable_persistent_cache=True
                )
                logger.info("✅ GNN cache manager initialized")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize GNN cache manager: {e}")
                gnn_cache_manager = None
        
        # Initialize GNN modules if available
        if HAS_GNN_MODULES and os.getenv("ENABLE_GNN_API", "true").lower() == "true":
            try:
                gnn_device = os.getenv("GNN_DEVICE", "auto")
                if gnn_device == "auto":
                    gnn_device = None
                
                if GNNEmbedder is not None:
                    gnn_embedder = GNNEmbedder(
                        embedding_dim=int(os.getenv("GNN_EMBEDDING_DIM", "128")),
                        hidden_dim=int(os.getenv("GNN_HIDDEN_DIM", "64")),
                        num_layers=int(os.getenv("GNN_NUM_LAYERS", "3")),
                        device=gnn_device
                    )
                    logger.info("✅ GNN Embedder initialized for API")
                    
                    # Pre-warm model to avoid first-request latency
                    if os.getenv("GNN_PREWARM", "true").lower() == "true":
                        try:
                            gnn_embedder.warm_up()
                        except Exception as e:
                            logger.warning(f"Model pre-warming failed (non-critical): {e}")
                
                if GNNNodeClassifier is not None:
                    gnn_classifier = GNNNodeClassifier(
                        hidden_dim=int(os.getenv("GNN_HIDDEN_DIM", "64")),
                        num_layers=int(os.getenv("GNN_NUM_LAYERS", "2")),
                        device=gnn_device
                    )
                    logger.info("✅ GNN Node Classifier initialized for API")
                
                if GNNLinkPredictor is not None:
                    gnn_link_predictor = GNNLinkPredictor(
                        hidden_dim=int(os.getenv("GNN_HIDDEN_DIM", "64")),
                        num_layers=int(os.getenv("GNN_NUM_LAYERS", "2")),
                        device=gnn_device
                    )
                    logger.info("✅ GNN Link Predictor initialized for API")
                
                if GNNAnomalyDetector is not None and os.getenv("ENABLE_GNN_ANOMALY_DETECTION", "false").lower() == "true":
                    gnn_anomaly_detector = GNNAnomalyDetector(device=gnn_device)
                    logger.info("✅ GNN Anomaly Detector initialized for API")
                
                if GNNSchemaMatcher is not None and os.getenv("ENABLE_GNN_SCHEMA_MATCHING", "false").lower() == "true":
                    gnn_schema_matcher = GNNSchemaMatcher(device=gnn_device)
                    logger.info("✅ GNN Schema Matcher initialized for API")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize GNN modules for API: {e}")
        
        logger.info("✅ Training Service initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Training Service: {e}")
        raise


# Request/Response Models
class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    components: Dict[str, bool]


class TrainingMetrics(BaseModel):
    epoch: int
    loss: float
    accuracy: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: float


class TrainingProgressResponse(BaseModel):
    training_run_id: str
    domain_id: Optional[str] = None
    status: str  # "running", "completed", "failed", "paused"
    current_epoch: int
    total_epochs: int
    metrics: List[TrainingMetrics]
    start_time: float
    elapsed_time: float
    estimated_time_remaining: Optional[float] = None


class ModelComparisonRequest(BaseModel):
    model_ids: List[str]
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "loss", "latency_ms"])


class ModelComparisonResponse(BaseModel):
    models: List[Dict[str, Any]]
    comparison: Dict[str, Any]
    rankings: Dict[str, List[Dict[str, Any]]]


class ExperimentRequest(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    status: str
    created_at: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PatternLearningRequest(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None
    glean_data: Optional[Dict[str, Any]] = None
    use_gnn: bool = False
    use_transformer: bool = False
    use_meta_patterns: bool = False
    use_active_learning: bool = False


class TrainingPipelineRequest(BaseModel):
    project_id: str
    system_id: Optional[str] = None
    json_tables: Optional[List[str]] = None
    hive_ddls: Optional[List[str]] = None
    control_m_files: Optional[List[str]] = None
    glean_days_back: int = 30
    enable_glean: bool = True
    enable_temporal_analysis: bool = True


class DomainTrainingRequest(BaseModel):
    domain_id: str
    training_data_path: str
    base_model_path: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    fine_tune: bool = True


class ABTestRequest(BaseModel):
    domain_id: str
    variant_a: Dict[str, Any]
    variant_b: Dict[str, Any]
    traffic_split: float = 0.5
    duration_days: int = 7


class ABTestRouteRequest(BaseModel):
    domain_id: str
    request_id: str


# Health Check
@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="training-service",
        timestamp=datetime.now().isoformat(),
        components={
            "pipeline": pipeline is not None,
            "pattern_engine": pattern_engine is not None,
            "domain_trainer": domain_trainer is not None,
            "metrics_collector": metrics_collector is not None,
            "ab_test_manager": ab_test_manager is not None,
            "gnn_available": HAS_GNN,
            "gnn_modules_available": HAS_GNN_MODULES,
            "gnn_api_enabled": gnn_embedder is not None,
            "gnn_classifier_available": gnn_classifier is not None,
            "gnn_link_predictor_available": gnn_link_predictor is not None,
            "meta_patterns_available": HAS_META_PATTERNS,
            "transformer_available": HAS_TRANSFORMER,
            "active_learning_available": HAS_ACTIVE_LEARNING,
            "pattern_transfer_available": HAS_PATTERN_TRANSFER,
            "auto_tuner_available": HAS_AUTO_TUNER,
        }
    )


# Pattern Learning Endpoints
@app.post("/patterns/learn")
async def learn_patterns(request: PatternLearningRequest):
    """Learn patterns from knowledge graph."""
    if not pattern_engine:
        raise HTTPException(status_code=503, detail="Pattern learning engine not initialized")
    
    try:
        # Create pattern engine with requested options
        engine = PatternLearningEngine(
            use_gnn=request.use_gnn,
            use_transformer=request.use_transformer
        )
        
        patterns = engine.learn_patterns(
            nodes=request.nodes,
            edges=request.edges,
            metrics=request.metrics,
            glean_data=request.glean_data
        )
        
        return {
            "status": "success",
            "patterns": patterns,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Pattern learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns/gnn/available")
async def gnn_available():
    """Check if GNN pattern learner is available."""
    return {
        "available": HAS_GNN,
        "status": "ready" if HAS_GNN else "not_available"
    }


@app.get("/patterns/meta/available")
async def meta_patterns_available():
    """Check if meta-pattern learner is available."""
    return {
        "available": HAS_META_PATTERNS,
        "status": "ready" if HAS_META_PATTERNS else "not_available"
    }


@app.get("/patterns/sequence/available")
async def sequence_patterns_available():
    """Check if sequence pattern transformer is available."""
    return {
        "available": HAS_TRANSFORMER,
        "status": "ready" if HAS_TRANSFORMER else "not_available"
    }


@app.get("/patterns/active/available")
async def active_patterns_available():
    """Check if active pattern learner is available."""
    return {
        "available": HAS_ACTIVE_LEARNING,
        "status": "ready" if HAS_ACTIVE_LEARNING else "not_available"
    }


@app.get("/patterns/transfer/available")
async def pattern_transfer_available():
    """Check if pattern transfer learner is available."""
    return {
        "available": HAS_PATTERN_TRANSFER,
        "status": "ready" if HAS_PATTERN_TRANSFER else "not_available"
    }


# Training Pipeline Endpoints
@app.post("/train/pipeline")
async def run_training_pipeline(request: TrainingPipelineRequest):
    """Run the complete training pipeline."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Training pipeline not initialized")
    
    try:
        results = pipeline.run_full_pipeline(
            project_id=request.project_id,
            system_id=request.system_id,
            json_tables=request.json_tables,
            hive_ddls=request.hive_ddls,
            control_m_files=request.control_m_files,
            glean_days_back=request.glean_days_back,
            enable_glean=request.enable_glean,
            enable_temporal_analysis=request.enable_temporal_analysis
        )
        
        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Domain Training Endpoints
@app.post("/train/domain")
async def train_domain_model(request: DomainTrainingRequest):
    """Train a domain-specific model."""
    if not domain_trainer:
        raise HTTPException(status_code=503, detail="Domain trainer not initialized")
    
    try:
        results = domain_trainer.train_domain_model(
            domain_id=request.domain_id,
            training_data_path=request.training_data_path,
            base_model_path=request.base_model_path,
            training_config=request.training_config,
            fine_tune=request.fine_tune
        )
        
        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Domain training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# A/B Testing Endpoints
@app.post("/ab-test/create")
async def create_ab_test(request: ABTestRequest):
    """Create an A/B test."""
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="A/B test manager not initialized")
    
    try:
        ab_test = ab_test_manager.create_ab_test(
            domain_id=request.domain_id,
            variant_a=request.variant_a,
            variant_b=request.variant_b,
            traffic_split=request.traffic_split,
            duration_days=request.duration_days
        )
        
        return {
            "status": "success",
            "ab_test": ab_test,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"A/B test creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ab-test/route")
async def route_ab_test(request: ABTestRouteRequest):
    """Route a request to A or B variant."""
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="A/B test manager not initialized")
    
    try:
        variant, config = ab_test_manager.route_request(
            domain_id=request.domain_id,
            request_id=request.request_id
        )
        
        return {
            "status": "success",
            "variant": variant,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"A/B test routing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Domain Metrics Endpoints
@app.get("/metrics/domain/{domain_id}")
async def get_domain_metrics(domain_id: str, time_window_days: int = 30):
    """Get domain performance metrics."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    
    try:
        metrics = metrics_collector.collect_domain_metrics(
            domain_id=domain_id,
            time_window_days=time_window_days
        )
        
        return {
            "status": "success",
            "domain_id": domain_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pattern Transfer Endpoints
class DomainSimilarityRequest(BaseModel):
    source_domain: str = Field(..., description="Source domain ID")
    target_domain: str = Field(..., description="Target domain ID")

@app.post("/patterns/transfer/calculate-similarity")
async def calculate_domain_similarity(
    request: DomainSimilarityRequest
):
    """Calculate domain similarity for pattern transfer."""
    if not HAS_PATTERN_TRANSFER:
        raise HTTPException(status_code=503, detail="Pattern transfer not available")
    
    try:
        from training.pattern_transfer import PatternTransferLearner
        transfer_learner = PatternTransferLearner()
        
        similarity = transfer_learner.calculate_domain_similarity(
            source_domain=request.source_domain,
            target_domain=request.target_domain
        )
        
        return {
            "status": "success",
            "source_domain": request.source_domain,
            "target_domain": request.target_domain,
            "similarity": similarity,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Domain similarity calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training Dashboard Endpoints
@app.get("/dashboard/progress")
async def get_training_progress(
    training_run_id: Optional[str] = Query(None),
    domain_id: Optional[str] = Query(None),
    include_history: bool = Query(True)
):
    """Get training progress with real-time metrics."""
    try:
        # Get training progress from domain trainer or pipeline
        progress_data = {
            "training_run_id": training_run_id or "unknown",
            "domain_id": domain_id,
            "status": "running",
            "current_epoch": 0,
            "total_epochs": 100,
            "metrics": [],
            "start_time": time.time() - 3600,  # Mock: 1 hour ago
            "elapsed_time": 3600.0,
            "estimated_time_remaining": 7200.0
        }
        
        # If domain trainer is available, get actual progress
        if domain_trainer and training_run_id:
            # In a real implementation, domain_trainer would track progress
            # For now, return mock data structure
            pass
        
        return TrainingProgressResponse(**progress_data)
    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dashboard/compare-models")
async def compare_models(request: ModelComparisonRequest):
    """Compare multiple models side-by-side."""
    try:
        models_data = []
        for model_id in request.model_ids:
            # Get model metrics (mock for now)
            model_data = {
                "model_id": model_id,
                "accuracy": 0.85,
                "loss": 0.25,
                "latency_ms": 150.0,
                "training_time": 3600.0,
                "created_at": datetime.now().isoformat()
            }
            models_data.append(model_data)
        
        # Generate comparison
        comparison = {
            "best_accuracy": max(m.get("accuracy", 0) for m in models_data),
            "best_latency": min(m.get("latency_ms", float('inf')) for m in models_data),
            "average_accuracy": sum(m.get("accuracy", 0) for m in models_data) / len(models_data) if models_data else 0
        }
        
        # Generate rankings
        rankings = {
            "accuracy": sorted(models_data, key=lambda x: x.get("accuracy", 0), reverse=True),
            "latency": sorted(models_data, key=lambda x: x.get("latency_ms", float('inf')))
        }
        
        return ModelComparisonResponse(
            models=models_data,
            comparison=comparison,
            rankings=rankings
        )
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/data-quality")
async def get_data_quality(domain_id: Optional[str] = None):
    """Get training data quality metrics."""
    try:
        # Mock data quality metrics
        quality_metrics = {
            "total_samples": 10000,
            "valid_samples": 9800,
            "invalid_samples": 200,
            "coverage": 0.95,
            "diversity_score": 0.87,
            "label_distribution": {
                "class_a": 0.4,
                "class_b": 0.35,
                "class_c": 0.25
            },
            "missing_values": 0.02,
            "outliers": 0.05
        }
        
        return {
            "status": "success",
            "domain_id": domain_id,
            "quality_metrics": quality_metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get data quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/patterns")
async def get_pattern_visualization(domain_id: Optional[str] = None):
    """Get pattern learning visualization data."""
    try:
        if not pattern_engine:
            raise HTTPException(status_code=503, detail="Pattern learning engine not initialized")
        
        # Get learned patterns (mock for now)
        patterns_data = {
            "column_patterns": [
                {"pattern": "VARCHAR -> TEXT", "frequency": 0.45, "confidence": 0.92},
                {"pattern": "INT -> BIGINT", "frequency": 0.32, "confidence": 0.88}
            ],
            "relationship_patterns": [
                {"pattern": "Table -> Column", "frequency": 0.78, "confidence": 0.95},
                {"pattern": "Column -> Column", "frequency": 0.22, "confidence": 0.82}
            ],
            "temporal_patterns": [
                {"pattern": "Schema evolution", "frequency": 0.15, "confidence": 0.75}
            ]
        }
        
        return {
            "status": "success",
            "domain_id": domain_id,
            "patterns": patterns_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get pattern visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/history")
async def get_training_history(
    domain_id: Optional[str] = None,
    limit: int = 20
):
    """Get training history with rollback support."""
    try:
        if not rollback_manager:
            raise HTTPException(status_code=503, detail="Rollback manager not initialized")
        
        # Get training history (mock for now)
        history = [
            {
                "training_run_id": f"run_{i}",
                "domain_id": domain_id or f"domain_{i % 3}",
                "status": "completed",
                "accuracy": 0.85 + (i * 0.01),
                "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                "can_rollback": i < 5  # Only recent runs can be rolled back
            }
            for i in range(limit)
        ]
        
        return {
            "status": "success",
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dashboard/rollback")
async def rollback_training(training_run_id: str):
    """Rollback to a previous training run."""
    try:
        if not rollback_manager:
            raise HTTPException(status_code=503, detail="Rollback manager not initialized")
        
        # Perform rollback
        result = rollback_manager.rollback_to_checkpoint(training_run_id)
        
        return {
            "status": "success",
            "training_run_id": training_run_id,
            "rollback_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to rollback training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Interactive Training Features
@app.post("/training/experiments")
async def create_experiment(request: ExperimentRequest):
    """Create a new training experiment."""
    try:
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store experiment metadata (in a real implementation, this would be persisted)
        experiment_data = {
            "experiment_id": experiment_id,
            "name": request.name,
            "description": request.description,
            "config": request.config,
            "tags": request.tags,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "metrics": {}
        }
        
        return ExperimentResponse(**experiment_data)
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/experiments")
async def list_experiments(
    domain_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20
):
    """List training experiments."""
    try:
        # Mock experiments list
        experiments = [
            {
                "experiment_id": f"exp_{i}",
                "name": f"Experiment {i}",
                "status": "completed" if i % 2 == 0 else "running",
                "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                "metrics": {
                    "accuracy": 0.85 + (i * 0.01),
                    "loss": 0.25 - (i * 0.01)
                }
            }
            for i in range(limit)
        ]
        
        if status:
            experiments = [e for e in experiments if e["status"] == status]
        
        return {
            "status": "success",
            "experiments": experiments,
            "count": len(experiments)
        }
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/sample-selection")
async def select_samples_for_active_learning(
    domain_id: str,
    num_samples: int = 10,
    strategy: str = "uncertainty"  # uncertainty, diversity, random
):
    """Select samples for active learning."""
    try:
        if not HAS_ACTIVE_LEARNING:
            raise HTTPException(status_code=503, detail="Active learning not available")
        
        # Mock sample selection
        selected_samples = [
            {
                "sample_id": f"sample_{i}",
                "uncertainty_score": 0.85 - (i * 0.05),
                "diversity_score": 0.7 + (i * 0.02),
                "priority": i + 1
            }
            for i in range(num_samples)
        ]
        
        return {
            "status": "success",
            "domain_id": domain_id,
            "strategy": strategy,
            "selected_samples": selected_samples,
            "count": len(selected_samples)
        }
    except Exception as e:
        logger.error(f"Failed to select samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/tune-parameters")
async def tune_training_parameters(
    domain_id: str,
    parameter_space: Dict[str, Any],
    optimization_goal: str = "accuracy"  # accuracy, latency, loss
):
    """Tune training parameters."""
    try:
        if not HAS_AUTO_TUNER:
            raise HTTPException(status_code=503, detail="Auto tuner not available")
        
        # Mock parameter tuning results
        best_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimization_goal": optimization_goal
        }
        
        return {
            "status": "success",
            "domain_id": domain_id,
            "best_parameters": best_params,
            "expected_improvement": 0.15,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to tune parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/pipeline-visualization")
async def get_pipeline_visualization():
    """Get training pipeline workflow visualization."""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Training pipeline not initialized")
        
        # Generate pipeline workflow visualization
        workflow = {
            "nodes": [
                {"id": "data_ingestion", "label": "Data Ingestion", "status": "completed"},
                {"id": "preprocessing", "label": "Preprocessing", "status": "completed"},
                {"id": "pattern_learning", "label": "Pattern Learning", "status": "running"},
                {"id": "model_training", "label": "Model Training", "status": "pending"},
                {"id": "evaluation", "label": "Evaluation", "status": "pending"},
                {"id": "deployment", "label": "Deployment", "status": "pending"}
            ],
            "edges": [
                {"from": "data_ingestion", "to": "preprocessing"},
                {"from": "preprocessing", "to": "pattern_learning"},
                {"from": "pattern_learning", "to": "model_training"},
                {"from": "model_training", "to": "evaluation"},
                {"from": "evaluation", "to": "deployment"}
            ],
            "current_step": "pattern_learning",
            "progress": 0.4
        }
        
        return {
            "status": "success",
            "workflow": workflow,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get pipeline visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "training-service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "patterns": {
                "learn": "/patterns/learn",
                "gnn": "/patterns/gnn/available",
                "meta": "/patterns/meta/available",
                "sequence": "/patterns/sequence/available",
                "active": "/patterns/active/available",
                "transfer": "/patterns/transfer/available"
            },
            "training": {
                "pipeline": "/train/pipeline",
                "domain": "/train/domain"
            },
            "dashboard": {
                "progress": "/dashboard/progress",
                "compare-models": "/dashboard/compare-models",
                "data-quality": "/dashboard/data-quality",
                "patterns": "/dashboard/patterns",
                "history": "/dashboard/history",
                "rollback": "/dashboard/rollback"
            },
            "interactive": {
                "experiments": "/training/experiments",
                "sample-selection": "/training/sample-selection",
                "tune-parameters": "/training/tune-parameters",
                "pipeline-visualization": "/training/pipeline-visualization"
            },
            "ab_testing": {
                "create": "/ab-test/create",
                "route": "/ab-test/route"
            },
            "metrics": {
                "domain": "/metrics/domain/{domain_id}"
            },
            "gnn": {
                "embeddings": "/gnn/embeddings",
                "classify": "/gnn/classify",
                "predict-links": "/gnn/predict-links",
                "structural-insights": "/gnn/structural-insights",
                "domain-model": "/gnn/domains/{domain_id}/model",
                "domain-query": "/gnn/domains/{domain_id}/query"
            }
        }
    }


# ============================================================================
# GNN API Endpoints (Priority 1: GNN Service API)
# ============================================================================

@app.post("/gnn/embeddings")
async def gnn_embeddings(request: GNNEmbeddingsRequest):
    """Generate GNN embeddings for nodes/graphs.
    
    This endpoint generates graph-level and/or node-level embeddings
    using the GNN embedder. Automatically detects domain if domain-specific
    models are available (Priority 6).
    """
    if not gnn_embedder:
        raise HTTPException(
            status_code=503,
            detail="GNN embedder not available. Ensure ENABLE_GNN_API=true and GNN modules are installed."
        )
    
    try:
        # Priority 6: Auto-detect domain and route if available
        domain_id = None
        if gnn_domain_router and request.nodes:
            domain_id = gnn_domain_router.detect_domain_from_nodes(request.nodes)
            if domain_id:
                logger.info(f"Auto-detected domain '{domain_id}' for embeddings request")
                # Route to domain-specific model if available
                _, domain_model_path = gnn_domain_router.route_to_domain_model(
                    domain_id,
                    "embeddings",
                    fallback_to_generic=True,
                )
                if domain_model_path:
                    logger.info(f"Using domain-specific embeddings model: {domain_model_path}")
                    # TODO: Load and use domain-specific model
        
        # Priority 7: Check cache first
        if gnn_cache_manager:
            cached_result = gnn_cache_manager.get(
                cache_type="embedding",
                nodes=request.nodes,
                edges=request.edges,
                domain_id=domain_id,
                config={"graph_level": request.graph_level}
            )
            if cached_result:
                logger.debug("Returning cached embeddings")
                return {
                    "status": "success",
                    "embeddings": cached_result,
                    "cached": True,
                    "timestamp": datetime.now().isoformat()
                }
        
        result = gnn_embedder.generate_embeddings(
            nodes=request.nodes,
            edges=request.edges,
            graph_level=request.graph_level
        )
        
        # Priority 7: Cache the result
        if gnn_cache_manager and "error" not in result:
            gnn_cache_manager.set(
                cache_type="embedding",
                data=result,
                nodes=request.nodes,
                edges=request.edges,
                domain_id=domain_id,
                config={"graph_level": request.graph_level},
                tags=["embedding", "gnn"]
            )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "embeddings": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"GNN embeddings generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gnn/classify")
async def gnn_classify(request: GNNClassifyRequest):
    """Classify nodes using GNN.
    
    This endpoint classifies graph nodes by type, domain, or quality
    using the GNN node classifier.
    """
    if not gnn_classifier:
        raise HTTPException(
            status_code=503,
            detail="GNN classifier not available. Ensure ENABLE_GNN_API=true and GNN modules are installed."
        )
    
    try:
        result = gnn_classifier.classify_nodes(
            nodes=request.nodes,
            edges=request.edges,
            top_k=request.top_k
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "classifications": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"GNN node classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gnn/predict-links")
async def gnn_predict_links(request: GNNPredictLinksRequest):
    """Predict missing relationships using GNN.
    
    This endpoint predicts missing links or suggests new relationships
    between nodes using the GNN link predictor.
    """
    if not gnn_link_predictor:
        raise HTTPException(
            status_code=503,
            detail="GNN link predictor not available. Ensure ENABLE_GNN_API=true and GNN modules are installed."
        )
    
    try:
        # Convert candidate_pairs from list of lists to list of tuples
        candidate_pairs_tuples = None
        if request.candidate_pairs:
            candidate_pairs_tuples = [
                (pair[0], pair[1]) if len(pair) >= 2 else (pair[0], "")
                for pair in request.candidate_pairs
            ]
        
        result = gnn_link_predictor.predict_links(
            nodes=request.nodes,
            edges=request.edges,
            candidate_pairs=candidate_pairs_tuples,
            top_k=request.top_k
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "predictions": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"GNN link prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gnn/structural-insights")
async def gnn_structural_insights(request: GNNStructuralInsightsRequest):
    """Get structural insights from graph using GNN.
    
    This endpoint provides structural insights including:
    - Anomaly detection
    - Pattern recognition
    - Structural analysis
    
    Results are cached for performance (Priority 7).
    """
    insights = {}
    
    try:
        # Priority 7: Check cache first
        if gnn_cache_manager:
            cached_result = gnn_cache_manager.get(
                cache_type="insight",
                nodes=request.nodes,
                edges=request.edges,
                config={
                    "insight_type": request.insight_type,
                    "threshold": request.threshold
                }
            )
            if cached_result:
                logger.debug("Returning cached structural insights")
                return {
                    "status": "success",
                    "insights": cached_result,
                    "cached": True,
                    "insight_type": request.insight_type,
                    "timestamp": datetime.now().isoformat()
                }
        # Anomaly detection
        if request.insight_type in ["anomalies", "all"] and gnn_anomaly_detector:
            try:
                anomaly_result = gnn_anomaly_detector.detect_anomalies(
                    nodes=request.nodes,
                    edges=request.edges,
                    threshold=request.threshold
                )
                if "error" not in anomaly_result:
                    insights["anomalies"] = anomaly_result
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")
                insights["anomalies"] = {"error": str(e)}
        
        # Pattern insights from embeddings
        if request.insight_type in ["patterns", "all"] and gnn_embedder:
            try:
                # Generate embeddings for pattern analysis
                embedding_result = gnn_embedder.generate_embeddings(
                    nodes=request.nodes,
                    edges=request.edges,
                    graph_level=False  # Get node-level embeddings for pattern analysis
                )
                if "error" not in embedding_result:
                    insights["patterns"] = {
                        "graph_embedding_dim": len(embedding_result.get("graph_embedding", [])),
                        "num_node_embeddings": len(embedding_result.get("node_embeddings", {})),
                        "embedding_available": True
                    }
            except Exception as e:
                logger.warning(f"Pattern analysis failed: {e}")
                insights["patterns"] = {"error": str(e)}
        
        # Classification insights
        if request.insight_type in ["patterns", "all"] and gnn_classifier:
            try:
                classify_result = gnn_classifier.classify_nodes(
                    nodes=request.nodes,
                    edges=request.edges
                )
                if "error" not in classify_result:
                    insights["node_types"] = {
                        "num_classified": len(classify_result.get("classifications", [])),
                        "num_classes": classify_result.get("num_classes", 0)
                    }
            except Exception as e:
                logger.warning(f"Classification insights failed: {e}")
        
        if not insights:
            raise HTTPException(
                status_code=503,
                detail="No GNN modules available for structural insights. Enable GNN modules and try again."
            )
        
        # Priority 7: Cache the result
        if gnn_cache_manager:
            gnn_cache_manager.set(
                cache_type="insight",
                data=insights,
                nodes=request.nodes,
                edges=request.edges,
                config={
                    "insight_type": request.insight_type,
                    "threshold": request.threshold
                },
                tags=["insight", "structural", "gnn"]
            )
        
        return {
            "status": "success",
            "insights": insights,
            "insight_type": request.insight_type,
            "cached": False,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GNN structural insights failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gnn/domains/{domain_id}/model")
async def get_domain_model(domain_id: str):
    """Get domain-specific GNN model information.
    
    Returns information about the domain-specific GNN model,
    including availability, model type, and capabilities.
    """
    try:
        # Priority 6: Use domain registry if available
        if gnn_domain_router:
            model_info = gnn_domain_router.get_domain_model_info(domain_id)
            return {
                "status": "success",
                "model": model_info,
                "timestamp": datetime.now().isoformat()
            }
        
        # Fallback to generic model info
        model_info = {
            "domain_id": domain_id,
            "models_available": False,
            "model_available": gnn_embedder is not None,
            "classifier_available": gnn_classifier is not None,
            "link_predictor_available": gnn_link_predictor is not None,
            "anomaly_detector_available": gnn_anomaly_detector is not None,
            "schema_matcher_available": gnn_schema_matcher is not None,
            "note": "Domain registry not available. Using generic models."
        }
        
        return {
            "status": "success",
            "model": model_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get domain model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gnn/domains/{domain_id}/query")
async def query_domain_gnn(domain_id: str, request: GNNDomainQueryRequest):
    """Query domain-specific GNN model.
    
    This endpoint routes queries to domain-specific GNN models
    based on the domain_id and query_type. If domain_id is "auto",
    domain is automatically detected from the query/nodes.
    """
    try:
        # Priority 6: Auto-detect domain if requested
        actual_domain_id = domain_id
        if domain_id == "auto" and gnn_domain_router:
            # Try to detect domain from nodes or query
            detected_domain = None
            if request.nodes:
                detected_domain = gnn_domain_router.detect_domain_from_nodes(request.nodes)
            
            if detected_domain:
                actual_domain_id = detected_domain
                logger.info(f"Auto-detected domain: {actual_domain_id}")
            else:
                logger.warning("Could not auto-detect domain, using generic models")
                actual_domain_id = None
        
        query_type = request.query_type
        
        # Priority 6: Route to domain-specific model if available
        model_type_map = {
            "embeddings": "embeddings",
            "classify": "classifier",
            "predict-links": "link_predictor",
            "insights": "anomaly_detector",
        }
        
        domain_model_path = None
        if actual_domain_id and gnn_domain_router:
            model_type = model_type_map.get(query_type, query_type)
            _, domain_model_path = gnn_domain_router.route_to_domain_model(
                actual_domain_id,
                model_type,
                fallback_to_generic=True,
            )
        
        # Use domain-specific model if available, otherwise use generic
        if domain_model_path:
            logger.info(f"Using domain-specific model: {actual_domain_id}/{query_type}")
            # TODO: Load and use domain-specific model from domain_model_path
            # For now, fall through to generic model
        
        if query_type == "embeddings":
            if not gnn_embedder:
                raise HTTPException(status_code=503, detail="GNN embedder not available")
            result = gnn_embedder.generate_embeddings(
                nodes=request.nodes,
                edges=request.edges,
                graph_level=request.query_params.get("graph_level", True) if request.query_params else True
            )
        
        elif query_type == "classify":
            if not gnn_classifier:
                raise HTTPException(status_code=503, detail="GNN classifier not available")
            result = gnn_classifier.classify_nodes(
                nodes=request.nodes,
                edges=request.edges,
                top_k=request.query_params.get("top_k") if request.query_params else None
            )
        
        elif query_type == "predict-links":
            if not gnn_link_predictor:
                raise HTTPException(status_code=503, detail="GNN link predictor not available")
            # Convert candidate_pairs if provided
            candidate_pairs = None
            if request.query_params and "candidate_pairs" in request.query_params:
                pairs = request.query_params["candidate_pairs"]
                if isinstance(pairs, list):
                    candidate_pairs = [
                        (pair[0], pair[1]) if isinstance(pair, list) and len(pair) >= 2 else (pair[0], "")
                        for pair in pairs
                    ]
            
            result = gnn_link_predictor.predict_links(
                nodes=request.nodes,
                edges=request.edges,
                candidate_pairs=candidate_pairs,
                top_k=request.query_params.get("top_k", 10) if request.query_params else 10
            )
        
        elif query_type == "insights":
            # Use structural insights endpoint logic
            threshold = request.query_params.get("threshold", 0.5) if request.query_params else 0.5
            insight_type = request.query_params.get("insight_type", "all") if request.query_params else "all"
            
            insights = {}
            if insight_type in ["anomalies", "all"] and gnn_anomaly_detector:
                anomaly_result = gnn_anomaly_detector.detect_anomalies(
                    nodes=request.nodes,
                    edges=request.edges,
                    threshold=threshold
                )
                if "error" not in anomaly_result:
                    insights["anomalies"] = anomaly_result
            
            if insight_type in ["patterns", "all"] and gnn_embedder:
                embedding_result = gnn_embedder.generate_embeddings(
                    nodes=request.nodes,
                    edges=request.edges,
                    graph_level=True
                )
                if "error" not in embedding_result:
                    insights["patterns"] = {
                        "graph_embedding_dim": len(embedding_result.get("graph_embedding", [])),
                        "embedding_available": True
                    }
            
            result = {"insights": insights}
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown query type: {query_type}. Supported: embeddings, classify, predict-links, insights"
            )
        
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "domain_id": actual_domain_id if actual_domain_id else domain_id,
            "query_type": query_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Domain GNN query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Priority 6: GNN Domain Registry API Endpoints

@app.post("/gnn/registry/register")
async def register_domain_model(request: RegisterModelRequest):
    """Register a domain-specific GNN model.
    
    Phase 3: Enhanced with model metadata (accuracy, training time, domain) for
    model sharing with graph service and A/B testing support.
    
    This endpoint registers a trained domain-specific GNN model
    in the registry for domain-aware routing.
    """
    if not gnn_domain_registry:
        raise HTTPException(
            status_code=503,
            detail="GNN domain registry not available. Ensure GNN domain routing is enabled."
        )
    
    try:
        # Phase 3: Extract additional metadata from training_metrics
        training_metrics = request.training_metrics or {}
        accuracy = training_metrics.get("accuracy", training_metrics.get("test_accuracy"))
        training_time = training_metrics.get("training_time_seconds", training_metrics.get("training_time"))
        
        model_info = gnn_domain_registry.register_model(
            domain_id=request.domain_id,
            model_type=request.model_type,
            model_path=request.model_path,
            version=request.version,
            training_metrics=request.training_metrics,
            model_config=request.model_config,
            description=request.description,
            is_active=request.is_active,
        )
        
        # Phase 3: Return enhanced metadata for graph service
        return {
            "status": "success",
            "model": {
                "domain_id": model_info.domain_id,
                "model_type": model_info.model_type,
                "version": model_info.version,
                "model_path": model_info.model_path,
                "created_at": model_info.created_at,
                "updated_at": model_info.updated_at,
                "is_active": model_info.is_active,
                "description": model_info.description,
                "accuracy": accuracy,
                "training_time_seconds": training_time,
                "has_metrics": bool(model_info.training_metrics),
                "has_config": bool(model_info.model_config),
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to register domain model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gnn/registry/domains")
async def list_domains():
    """List all domains with registered GNN models."""
    if not gnn_domain_registry:
        raise HTTPException(
            status_code=503,
            detail="GNN domain registry not available."
        )
    
    try:
        domains = gnn_domain_registry.list_domains()
        return {
            "status": "success",
            "domains": domains,
            "count": len(domains),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gnn/registry/domains/{domain_id}/models")
async def list_domain_models(domain_id: str, active_only: bool = True):
    """List all models for a specific domain.
    
    Phase 3: Enhanced to include model metadata for graph service integration.
    """
    if not gnn_domain_registry:
        raise HTTPException(
            status_code=503,
            detail="GNN domain registry not available."
        )
    
    try:
        models = gnn_domain_registry.list_models_for_domain(domain_id, active_only=active_only)
        models_info = {}
        for model_type, model_info in models.items():
            # Phase 3: Extract metadata for graph service
            training_metrics = model_info.training_metrics or {}
            accuracy = training_metrics.get("accuracy", training_metrics.get("test_accuracy"))
            training_time = training_metrics.get("training_time_seconds", training_metrics.get("training_time"))
            
            models_info[model_type] = {
                "version": model_info.version,
                "model_path": model_info.model_path,
                "created_at": model_info.created_at,
                "updated_at": model_info.updated_at,
                "description": model_info.description,
                "is_active": model_info.is_active,
                "accuracy": accuracy,
                "training_time_seconds": training_time,
                "has_metrics": bool(model_info.training_metrics),
                "has_config": bool(model_info.model_config),
            }
        
        return {
            "status": "success",
            "domain_id": domain_id,
            "models": models_info,
            "count": len(models_info),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list domain models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gnn/registry/models/{domain_id}/{model_type}")
async def get_model_info(domain_id: str, model_type: str):
    """Get information about a specific model for graph service.
    
    Phase 3: Model serving endpoint for graph service to query model information.
    """
    if not gnn_domain_registry:
        raise HTTPException(
            status_code=503,
            detail="GNN domain registry not available."
        )
    
    try:
        models = gnn_domain_registry.list_models_for_domain(domain_id, active_only=False)
        
        if model_type not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_type} not found for domain {domain_id}"
            )
        
        model_info = models[model_type]
        training_metrics = model_info.training_metrics or {}
        accuracy = training_metrics.get("accuracy", training_metrics.get("test_accuracy"))
        training_time = training_metrics.get("training_time_seconds", training_metrics.get("training_time"))
        
        return {
            "status": "success",
            "model": {
                "domain_id": model_info.domain_id,
                "model_type": model_info.model_type,
                "version": model_info.version,
                "model_path": model_info.model_path,
                "created_at": model_info.created_at,
                "updated_at": model_info.updated_at,
                "description": model_info.description,
                "is_active": model_info.is_active,
                "accuracy": accuracy,
                "training_time_seconds": training_time,
                "training_metrics": model_info.training_metrics,
                "model_config": model_info.model_config,
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Priority 7: GNN Cache Management API Endpoints

@app.get("/gnn/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    if not gnn_cache_manager:
        raise HTTPException(
            status_code=503,
            detail="GNN cache manager not available."
        )
    
    try:
        stats = gnn_cache_manager.get_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gnn/cache/invalidate")
async def invalidate_cache(
    cache_type: Optional[str] = None,
    domain_id: Optional[str] = None,
    tags: Optional[str] = None
):
    """Invalidate cache entries.
    
    Query parameters:
    - cache_type: Invalidate entries of specific type (embedding, insight, etc.)
    - domain_id: Invalidate entries for specific domain
    - tags: Comma-separated list of tags to invalidate
    """
    if not gnn_cache_manager:
        raise HTTPException(
            status_code=503,
            detail="GNN cache manager not available."
        )
    
    try:
        tag_list = tags.split(",") if isinstance(tags, str) else tags
        invalidated = gnn_cache_manager.invalidate(
            cache_type=cache_type,
            domain_id=domain_id,
            tags=tag_list
        )
        
        return {
            "status": "success",
            "invalidated": invalidated,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/gnn/cache/clear")
async def clear_cache():
    """Clear all cache entries."""
    if not gnn_cache_manager:
        raise HTTPException(
            status_code=503,
            detail="GNN cache manager not available."
        )
    
    try:
        gnn_cache_manager.clear()
        return {
            "status": "success",
            "message": "Cache cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Training Service on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )

