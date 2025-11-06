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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

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


@app.on_event("startup")
async def startup_event():
    """Initialize service components on startup."""
    global pipeline, pattern_engine, domain_trainer, metrics_collector
    global ab_test_manager, rollback_manager, routing_optimizer, domain_optimizer, domain_filter
    
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
            "ab_testing": {
                "create": "/ab-test/create",
                "route": "/ab-test/route"
            },
            "metrics": {
                "domain": "/metrics/domain/{domain_id}"
            }
        }
    }


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

