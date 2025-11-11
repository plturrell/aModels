"""Training pipeline orchestration.

This module provides end-to-end training pipeline that integrates:
1. Extract service for knowledge graph generation
2. Glean Catalog for historical patterns
3. Pattern learning algorithms
4. Model training
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime

from .glean_integration import GleanTrainingClient, ingest_glean_data_for_training
from .pattern_learning import PatternLearningEngine, WorkflowPatternLearner
from .temporal_analysis import TemporalPatternLearner, SchemaEvolutionAnalyzer
from .domain_filter import DomainFilter, PrivacyConfig
from .domain_trainer import DomainTrainer
from .domain_metrics import DomainMetricsCollector
from .ab_testing import ABTestManager
from .rollback_manager import RollbackManager
from .routing_optimizer import RoutingOptimizer
from .domain_optimizer import DomainOptimizer
from .digital_twin import DigitalTwinSimulator
from .langsmith_tracing import LangSmithTracer
from .graph_client import GraphServiceClient
from .data_access import UnifiedDataAccess

# Coral NPU integration
try:
    from .coralnpu_client import CoralNPUClient, create_coralnpu_client
    from .coralnpu_quantization import CoralNPUQuantizer, create_quantizer
    from .coralnpu_converter import CoralNPUConverter, create_converter
    HAS_CORALNPU = True
except ImportError:
    HAS_CORALNPU = False
    CoralNPUClient = None
    CoralNPUQuantizer = None
    CoralNPUConverter = None
    create_coralnpu_client = None
    create_quantizer = None
    create_converter = None

# Phase 9.1: Auto-tuning
try:
    from .auto_tuner import AutoTuner
    HAS_AUTO_TUNER = True
except ImportError:
    HAS_AUTO_TUNER = False
    AutoTuner = None

# GNN modules (Priority 1: Integration)
try:
    from .gnn_embeddings import GNNEmbedder
    from .gnn_node_classifier import GNNNodeClassifier
    from .gnn_link_predictor import GNNLinkPredictor
    from .gnn_anomaly_detector import GNNAnomalyDetector
    from .gnn_schema_matcher import GNNSchemaMatcher
    HAS_GNN_MODULES = True
except ImportError:
    HAS_GNN_MODULES = False
    GNNEmbedder = None
    GNNNodeClassifier = None
    GNNLinkPredictor = None
    GNNAnomalyDetector = None
    GNNSchemaMatcher = None

# GNN Training and Evaluation (Priority 2)
try:
    from .gnn_training import GNNTrainer
    from .gnn_evaluation import GNNEvaluator
    HAS_GNN_TRAINING = True
except ImportError:
    HAS_GNN_TRAINING = False
    GNNTrainer = None
    GNNEvaluator = None

# GNN Priority 3: Advanced Features
try:
    from .gnn_multimodal import MultiModalGNN
    from .gnn_hyperparameter_tuning import GNNHyperparameterTuner
    from .gnn_cross_validation import GNNCrossValidator
    from .gnn_ensembling import GNNEnsemble, GNNEnsembleBuilder
    from .gnn_transfer_learning import GNNTransferLearner
    from .gnn_active_learning import GNNActiveLearner
    HAS_GNN_PRIORITY3 = True
except ImportError:
    HAS_GNN_PRIORITY3 = False
    MultiModalGNN = None
    GNNHyperparameterTuner = None
    GNNCrossValidator = None
    GNNEnsemble = None
    GNNEnsembleBuilder = None
    GNNTransferLearner = None
    GNNActiveLearner = None

# GNN Priority 4: Performance Optimization
try:
    from .gnn_batch_processing import EmbeddingCache, GraphBatchProcessor, MemoryOptimizer
    from .gnn_model_optimization import (
        GNNModelQuantizer, GNNModelPruner, GNNInferenceOptimizer, GNNModelOptimizer
    )
    HAS_GNN_PRIORITY4 = True
except ImportError:
    HAS_GNN_PRIORITY4 = False
    EmbeddingCache = None
    GraphBatchProcessor = None
    MemoryOptimizer = None
    GNNModelQuantizer = None
    GNNModelPruner = None
    GNNInferenceOptimizer = None
    GNNModelOptimizer = None

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline orchestrator."""
    
    def __init__(
        self,
        extract_service_url: Optional[str] = None,
        graph_service_url: Optional[str] = None,
        glean_db_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        enable_domain_filtering: bool = True,
        privacy_level: str = "medium",
        enable_gnn: Optional[bool] = None,
        enable_gnn_embeddings: Optional[bool] = None,
        enable_gnn_classification: Optional[bool] = None,
        enable_gnn_link_prediction: Optional[bool] = None,
        enable_gnn_device: Optional[str] = None,
        enable_gnn_training: Optional[bool] = None,
        gnn_models_dir: Optional[str] = None,
        enable_coralnpu: Optional[bool] = None,
        coralnpu_quantize: Optional[bool] = None,
        coralnpu_compile: Optional[bool] = None,
    ):
        self.extract_service_url = extract_service_url or os.getenv("EXTRACT_SERVICE_URL", "http://localhost:19080")
        # Phase 1: Initialize Graph service client for optimized Neo4j access
        self.graph_service_url = graph_service_url or os.getenv("GRAPH_SERVICE_URL")
        self.graph_client = None
        if os.getenv("ENABLE_GRAPH_SERVICE_INTEGRATION", "true").lower() == "true":
            try:
                self.graph_client = GraphServiceClient(
                    graph_service_url=self.graph_service_url,
                    extract_service_url=self.extract_service_url
                )
                logger.info("Graph service client initialized for optimized Neo4j access")
            except Exception as e:
                logger.warning(f"Failed to initialize Graph service client (falling back to Extract service): {e}")
                self.graph_client = None
        
        # Improvement 4: Initialize unified data access layer
        self.unified_data_access = None
        if os.getenv("ENABLE_UNIFIED_DATA_ACCESS", "true").lower() == "true":
            try:
                from .gnn_cache_manager import GNNCacheManager
                cache_manager = GNNCacheManager(
                    redis_url=os.getenv("REDIS_URL"),
                    default_ttl=int(os.getenv("GNN_CACHE_TTL", "3600"))
                )
                self.unified_data_access = UnifiedDataAccess(
                    postgres_dsn=os.getenv("POSTGRES_DSN"),
                    redis_url=os.getenv("REDIS_URL"),
                    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
                    neo4j_password=os.getenv("NEO4J_PASSWORD", "amodels123"),
                    cache_manager=cache_manager
                )
                logger.info("Unified data access layer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize unified data access layer: {e}")
                self.unified_data_access = None
        
        self.glean_client = GleanTrainingClient(db_name=glean_db_name)
        self.output_dir = output_dir or os.getenv("TRAINING_OUTPUT_DIR", "./training_data")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize domain filter with differential privacy
        self.enable_domain_filtering = enable_domain_filtering
        if self.enable_domain_filtering:
            privacy_config = PrivacyConfig(privacy_level=privacy_level)
            self.domain_filter = DomainFilter(
                localai_url=os.getenv("LOCALAI_URL", "http://localai:8080"),
                privacy_config=privacy_config
            )
        else:
            self.domain_filter = None
        
        # Initialize domain trainer and metrics collector
        self.domain_trainer = DomainTrainer(
            localai_url=os.getenv("LOCALAI_URL", "http://localai:8080"),
            postgres_dsn=os.getenv("POSTGRES_DSN"),
            redis_url=os.getenv("REDIS_URL")
        )
        self.metrics_collector = DomainMetricsCollector(
            localai_url=os.getenv("LOCALAI_URL", "http://localai:8080"),
            postgres_dsn=os.getenv("POSTGRES_DSN")
        )
        
        # Initialize Phase 3 components
        self.ab_test_manager = ABTestManager(
            postgres_dsn=os.getenv("POSTGRES_DSN"),
            redis_url=os.getenv("REDIS_URL")
        )
        self.rollback_manager = RollbackManager(
            postgres_dsn=os.getenv("POSTGRES_DSN"),
            redis_url=os.getenv("REDIS_URL"),
            localai_url=os.getenv("LOCALAI_URL", "http://localai:8080")
        )
        self.routing_optimizer = RoutingOptimizer(
            postgres_dsn=os.getenv("POSTGRES_DSN"),
            learning_rate=float(os.getenv("ROUTING_LEARNING_RATE", "0.1"))
        )
        self.domain_optimizer = DomainOptimizer(
            redis_url=os.getenv("REDIS_URL"),
            cache_ttl=int(os.getenv("DOMAIN_CACHE_TTL", "3600"))
        )

        self.digital_twin = DigitalTwinSimulator(logger=logger)
        self.langsmith_tracer = LangSmithTracer(logger=logger)

        # Phase 9.1: Initialize auto-tuner if enabled
        self.auto_tuner = None
        if (
            os.getenv("ENABLE_AUTO_TUNING", "false").lower() == "true"
            and HAS_AUTO_TUNER
            and AutoTuner is not None
        ):
            try:
                self.auto_tuner = AutoTuner(
                    study_name=os.getenv("OPTUNA_STUDY_NAME", "amodels_training"),
                    storage=os.getenv("OPTUNA_STORAGE"),
                    n_trials=int(os.getenv("OPTUNA_N_TRIALS", "50")),
                )
                logger.info("Auto-tuner initialized (Phase 9.1)")
            except Exception as e:
                logger.warning(f"Failed to initialize auto-tuner: {e}")
                self.auto_tuner = None
        
        # Priority 1: Initialize GNN modules if enabled
        self.enable_gnn = enable_gnn if enable_gnn is not None else os.getenv("ENABLE_GNN", "false").lower() == "true"
        self.enable_gnn_embeddings = enable_gnn_embeddings if enable_gnn_embeddings is not None else os.getenv("ENABLE_GNN_EMBEDDINGS", "true").lower() == "true"
        self.enable_gnn_classification = enable_gnn_classification if enable_gnn_classification is not None else os.getenv("ENABLE_GNN_CLASSIFICATION", "true").lower() == "true"
        self.enable_gnn_link_prediction = enable_gnn_link_prediction if enable_gnn_link_prediction is not None else os.getenv("ENABLE_GNN_LINK_PREDICTION", "true").lower() == "true"
        
        self.gnn_embedder = None
        self.gnn_classifier = None
        self.gnn_link_predictor = None
        self.gnn_anomaly_detector = None
        self.gnn_schema_matcher = None
        
        if self.enable_gnn and HAS_GNN_MODULES:
            try:
                gnn_device = enable_gnn_device or os.getenv("GNN_DEVICE", "auto")
                if gnn_device == "auto":
                    gnn_device = None  # Let modules auto-detect
                
                if self.enable_gnn_embeddings and GNNEmbedder is not None:
                    self.gnn_embedder = GNNEmbedder(
                        embedding_dim=int(os.getenv("GNN_EMBEDDING_DIM", "128")),
                        hidden_dim=int(os.getenv("GNN_HIDDEN_DIM", "64")),
                        num_layers=int(os.getenv("GNN_NUM_LAYERS", "3")),
                        device=gnn_device
                    )
                    logger.info("GNN Embedder initialized")
                
                if self.enable_gnn_classification and GNNNodeClassifier is not None:
                    self.gnn_classifier = GNNNodeClassifier(
                        hidden_dim=int(os.getenv("GNN_HIDDEN_DIM", "64")),
                        num_layers=int(os.getenv("GNN_NUM_LAYERS", "2")),
                        device=gnn_device
                    )
                    logger.info("GNN Node Classifier initialized")
                
                if self.enable_gnn_link_prediction and GNNLinkPredictor is not None:
                    self.gnn_link_predictor = GNNLinkPredictor(
                        hidden_dim=int(os.getenv("GNN_HIDDEN_DIM", "64")),
                        num_layers=int(os.getenv("GNN_NUM_LAYERS", "2")),
                        device=gnn_device
                    )
                    logger.info("GNN Link Predictor initialized")
                
                # Optional: Anomaly detector and schema matcher
                if os.getenv("ENABLE_GNN_ANOMALY_DETECTION", "false").lower() == "true" and GNNAnomalyDetector is not None:
                    self.gnn_anomaly_detector = GNNAnomalyDetector(device=gnn_device)
                    logger.info("GNN Anomaly Detector initialized")
                
                if os.getenv("ENABLE_GNN_SCHEMA_MATCHING", "false").lower() == "true" and GNNSchemaMatcher is not None:
                    self.gnn_schema_matcher = GNNSchemaMatcher(device=gnn_device)
                    logger.info("GNN Schema Matcher initialized")
                
                logger.info("GNN modules initialized (Priority 1: Integration)")
            except Exception as e:
                logger.warning(f"Failed to initialize GNN modules: {e}")
                self.enable_gnn = False
        else:
            if not HAS_GNN_MODULES:
                logger.info("GNN modules not available (PyTorch Geometric not installed)")
            else:
                logger.info("GNN modules disabled")
        
        # Priority 2: Initialize GNN trainer if enabled
        self.enable_gnn_training = enable_gnn_training if enable_gnn_training is not None else os.getenv("ENABLE_GNN_TRAINING", "false").lower() == "true"
        self.gnn_trainer = None
        
        if self.enable_gnn_training and HAS_GNN_TRAINING and GNNTrainer is not None:
            try:
                gnn_models_dir = gnn_models_dir or os.getenv("GNN_MODELS_DIR", os.path.join(self.output_dir, "gnn_models"))
                self.gnn_trainer = GNNTrainer(
                    output_dir=gnn_models_dir,
                    device=enable_gnn_device or os.getenv("GNN_DEVICE", "auto")
                )
                logger.info("GNN Trainer initialized (Priority 2: Training)")
            except Exception as e:
                logger.warning(f"Failed to initialize GNN trainer: {e}")
                self.enable_gnn_training = False
        
        # Priority 4: Initialize embedding cache
        self.enable_gnn_cache = False
        self.embedding_cache = None
        if HAS_GNN_PRIORITY4:
            try:
                self.enable_gnn_cache = os.getenv("ENABLE_GNN_CACHE", "true").lower() == "true"
                if self.enable_gnn_cache:
                    cache_dir = os.getenv("GNN_CACHE_DIR", os.path.join(self.output_dir, "gnn_cache"))
                    max_size = int(os.getenv("GNN_CACHE_MAX_SIZE", "1000"))
                    self.embedding_cache = EmbeddingCache(cache_dir=cache_dir, max_size=max_size)
                    logger.info("GNN EmbeddingCache initialized (Priority 4: Performance)")
            except Exception as e:
                logger.warning(f"Failed to initialize EmbeddingCache: {e}")

        # Priority 4: Initialize inference optimizer
        self.enable_gnn_inference_opt = False
        self.gnn_inference_optimizer = None
        self._embedder_optimized = False
        self._classifier_optimized = False
        self._link_predictor_optimized = False
        if HAS_GNN_PRIORITY4:
            try:
                self.enable_gnn_inference_opt = os.getenv("ENABLE_GNN_INFERENCE_OPT", "true").lower() == "true"
                if self.enable_gnn_inference_opt:
                    use_jit = os.getenv("GNN_INFERENCE_USE_JIT", "true").lower() == "true"
                    use_compile = os.getenv("GNN_INFERENCE_USE_TORCH_COMPILE", "true").lower() == "true"
                    self.gnn_inference_optimizer = GNNInferenceOptimizer(use_jit=use_jit, use_torch_compile=use_compile)
                    logger.info("GNN Inference optimizer initialized (Priority 4: Performance)")
            except Exception as e:
                logger.warning(f"Failed to initialize GNN inference optimizer: {e}")
        
        # Initialize Coral NPU integration
        self.enable_coralnpu = enable_coralnpu if enable_coralnpu is not None else (
            os.getenv("CORALNPU_ENABLED", "false").lower() == "true"
        )
        self.coralnpu_quantize = coralnpu_quantize if coralnpu_quantize is not None else (
            os.getenv("CORALNPU_QUANTIZE_MODELS", "false").lower() == "true"
        )
        self.coralnpu_compile = coralnpu_compile if coralnpu_compile is not None else (
            os.getenv("CORALNPU_COMPILE_MODELS", "false").lower() == "true"
        )
        
        self.coralnpu_client = None
        self.coralnpu_quantizer = None
        self.coralnpu_converter = None
        
        if self.enable_coralnpu and HAS_CORALNPU:
            try:
                # Create metrics collector for Coral NPU
                def coralnpu_metrics_collector(service: str, operation: str, latency: float, is_npu: bool):
                    device = "npu" if is_npu else "cpu"
                    logger.info(f"Coral NPU {operation}: {latency:.3f}s on {device}")
                
                self.coralnpu_client = create_coralnpu_client(
                    enabled=self.enable_coralnpu,
                    fallback_to_cpu=True,
                    metrics_collector=coralnpu_metrics_collector,
                )
                
                if self.coralnpu_quantize:
                    self.coralnpu_quantizer = create_quantizer(
                        client=self.coralnpu_client,
                        enabled=True,
                    )
                
                self.coralnpu_converter = create_converter(
                    client=self.coralnpu_client,
                    quantizer=self.coralnpu_quantizer,
                )
                
                logger.info("Coral NPU integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Coral NPU: {e}")
                self.enable_coralnpu = False
        else:
            if not HAS_CORALNPU:
                logger.info("Coral NPU modules not available")
            else:
                logger.info("Coral NPU integration disabled")
    
    def run_full_pipeline(
        self,
        project_id: str,
        system_id: Optional[str] = None,
        json_tables: Optional[list] = None,
        hive_ddls: Optional[list] = None,
        control_m_files: Optional[list] = None,
        signavio_files: Optional[list] = None,
        glean_days_back: int = 30,
        enable_glean: bool = True,
        enable_temporal_analysis: bool = True,
        enable_digital_twin: bool = True,
        train_gnn_models: bool = False,
        gnn_training_epochs: int = 100,
    ) -> Dict[str, Any]:
        """Run the complete training pipeline.
        
        Steps:
        1. Extract knowledge graph from source data
        2. Query Glean for historical patterns (if enabled)
        3. Generate training features
        4. Prepare training dataset
        5. Return training data summary
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
            json_tables: List of JSON table file paths
            hive_ddls: List of Hive DDL file paths
            control_m_files: List of Control-M XML file paths
            signavio_files: List of Signavio BPMN/JSON exports to merge into the graph
            glean_days_back: Number of days to look back in Glean
            enable_glean: Whether to enable Glean integration
            enable_digital_twin: Whether to execute the digital twin simulation hook
        
        Returns:
            Dictionary with training pipeline results
        """
        logger.info(f"Starting training pipeline for project={project_id}, system={system_id}")
        
        results = {
            "project_id": project_id,
            "system_id": system_id,
            "pipeline_started_at": datetime.now().isoformat(),
            "steps": {},
        }

        results["steps"]["signavio_ingest"] = {"status": "skipped"}
        
        # Priority 2: Train GNN models if requested (before extraction to use for inference)
        if train_gnn_models and self.enable_gnn_training and self.gnn_trainer:
            logger.info("Priority 2: Training GNN models before pipeline execution...")
            try:
                # Phase 6: Use graph service streaming for efficient training data loading
                training_nodes = []
                training_edges = []
                
                if self.graph_client:
                    # Use graph service client streaming for optimized batch loading
                    logger.info("Using graph service streaming for GNN training data...")
                    batch_size = int(os.getenv("GNN_TRAINING_BATCH_SIZE", "1000"))
                    
                    # Stream nodes
                    for node_batch in self.graph_client.stream_nodes(
                        project_id=project_id,
                        system_id=system_id,
                        batch_size=batch_size
                    ):
                        training_nodes.extend([node.dict() for node in node_batch])
                    
                    # Stream edges
                    for edge_batch in self.graph_client.stream_edges(
                        project_id=project_id,
                        system_id=system_id,
                        batch_size=batch_size
                    ):
                        training_edges.extend([edge.dict() for edge in edge_batch])
                    
                    logger.info(f"Streamed {len(training_nodes)} nodes, {len(training_edges)} edges from graph service")
                else:
                    # Fallback to extract service
                    initial_graph = self._extract_knowledge_graph(
                        project_id=project_id,
                        system_id=system_id,
                        json_tables=json_tables or [],
                        hive_ddls=hive_ddls or [],
                        control_m_files=control_m_files or [],
                        signavio_files=signavio_files or [],
                    )
                    
                    training_nodes = initial_graph.get("nodes", [])
                    training_edges = initial_graph.get("edges", [])
                
                if training_nodes and training_edges:
                    training_results = self.train_gnn_models(
                        training_nodes,
                        training_edges,
                        train_classifier=True,
                        train_link_predictor=True,
                        epochs=gnn_training_epochs
                    )
                    results["steps"]["gnn_training"] = training_results
                    
                    # Load trained models into GNN modules
                    if training_results.get("status") == "success":
                        model_paths = {}
                        if "classifier" in training_results.get("training_results", {}):
                            classifier_path = training_results["training_results"]["classifier"].get("model_path")
                            if classifier_path and self.gnn_classifier:
                                try:
                                    self.gnn_classifier.load_model(classifier_path)
                                    logger.info("✅ Loaded trained classifier into pipeline")
                                except Exception as e:
                                    logger.warning(f"Failed to load classifier: {e}")
                        
                        if "link_predictor" in training_results.get("training_results", {}):
                            predictor_path = training_results["training_results"]["link_predictor"].get("model_path")
                            if predictor_path and self.gnn_link_predictor:
                                try:
                                    self.gnn_link_predictor.load_model(predictor_path)
                                    logger.info("✅ Loaded trained link predictor into pipeline")
                                except Exception as e:
                                    logger.warning(f"Failed to load link predictor: {e}")
                else:
                    results["steps"]["gnn_training"] = {
                        "status": "skipped",
                        "reason": "No graph data available for training"
                    }
            except Exception as e:
                logger.warning(f"⚠️  GNN training failed (continuing): {e}")
                results["steps"]["gnn_training"] = {"status": "failed", "error": str(e)}
        else:
            results["steps"]["gnn_training"] = {"status": "skipped", "reason": "not requested or not available"}
        
        # Step 1: Extract knowledge graph from source data
        logger.info("Step 1: Extracting knowledge graph from source data...")
        signavio_summary = None
        try:
            graph_data = self._extract_knowledge_graph(
                project_id=project_id,
                system_id=system_id,
                json_tables=json_tables or [],
                hive_ddls=hive_ddls or [],
                control_m_files=control_m_files or [],
                signavio_files=signavio_files or [],
            )
            results["steps"]["extract"] = {
                "status": "success",
                "nodes": len(graph_data.get("nodes", [])),
                "edges": len(graph_data.get("edges", [])),
            }
            signavio_info = graph_data.get("signavio") if isinstance(graph_data, dict) else None
            service_unavailable = bool(graph_data.get("service_unavailable"))
            if isinstance(signavio_info, dict):
                results["steps"]["signavio_ingest"] = {
                    "status": "success",
                    "processes": signavio_info.get("process_count", 0),
                    "source_files": signavio_info.get("source_files", 0),
                }
                signavio_summary = signavio_info
                if service_unavailable:
                    results["steps"]["signavio_ingest"]["status"] = "failed"
                    results["steps"]["signavio_ingest"]["error"] = signavio_info.get(
                        "error", "extract service unavailable"
                    )
                    logger.warning(
                        "Signavio ingestion skipped: %s",
                        results["steps"]["signavio_ingest"]["error"],
                    )
            elif signavio_files:
                message = "extract service unavailable" if service_unavailable else "extract service returned no Signavio data"
                results["steps"]["signavio_ingest"] = {
                    "status": "failed",
                    "processes": 0,
                    "source_files": len([f for f in signavio_files if f]),
                    "error": message,
                }
                logger.warning("Signavio ingestion skipped: %s", message)
            logger.info(f"✅ Extracted {results['steps']['extract']['nodes']} nodes, {results['steps']['extract']['edges']} edges")
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            results["steps"]["extract"] = {"status": "failed", "error": str(e)}
            return results
        
        # Step 2: Query Glean for historical patterns
        glean_data = None
        if enable_glean:
            logger.info("Step 2: Querying Glean Catalog for historical patterns...")
            try:
                glean_data = ingest_glean_data_for_training(
                    project_id=project_id,
                    system_id=system_id,
                    days_back=glean_days_back,
                    output_dir=os.path.join(self.output_dir, "glean")
                )
                results["steps"]["glean"] = {
                    "status": "success",
                    "nodes": glean_data["metadata"]["node_count"],
                    "edges": glean_data["metadata"]["edge_count"],
                    "metrics_available": bool(glean_data.get("metrics")),
                }
                logger.info(f"✅ Ingested {results['steps']['glean']['nodes']} nodes, {results['steps']['glean']['edges']} edges from Glean")
            except Exception as e:
                logger.warning(f"⚠️  Glean ingestion failed (continuing without historical data): {e}")
                results["steps"]["glean"] = {"status": "failed", "error": str(e)}
        else:
            results["steps"]["glean"] = {"status": "skipped"}

        # Initialize optional semantic embeddings variable used later
        semantic_embeddings = None
        
        # Step 3: Learn patterns from graph and Glean data
        logger.info("Step 3: Learning patterns from knowledge graph and Glean data...")
        try:
            pattern_engine = PatternLearningEngine()
            
            # Extract nodes and edges from graph data
            graph_nodes = graph_data.get("nodes", [])
            graph_edges = graph_data.get("edges", [])
            graph_metrics = graph_data.get("metrics", {})
            
            # Extract nodes and edges from Glean data if available
            glean_nodes = glean_data.get("nodes", []) if glean_data else []
            glean_edges = glean_data.get("edges", []) if glean_data else []
            
            # Combine current and historical data
            all_nodes = graph_nodes + glean_nodes
            all_edges = graph_edges + glean_edges
            
            # Learn patterns (including semantic patterns if available)
            learned_patterns = pattern_engine.learn_patterns(
                nodes=all_nodes,
                edges=all_edges,
                metrics=graph_metrics,
                glean_data=glean_data,
                semantic_embeddings=semantic_embeddings
            )
            
            results["steps"]["pattern_learning"] = {
                "status": "success",
                "column_patterns": learned_patterns["column_patterns"].get("unique_types", 0),
                "relationship_patterns": learned_patterns["relationship_patterns"].get("unique_labels", 0),
                "metrics_patterns": bool(learned_patterns.get("metrics_patterns")),
            }
            logger.info(f"✅ Learned patterns: {results['steps']['pattern_learning']['column_patterns']} column types, {results['steps']['pattern_learning']['relationship_patterns']} relationship types")
        except Exception as e:
            logger.error(f"❌ Pattern learning failed: {e}")
            results["steps"]["pattern_learning"] = {"status": "failed", "error": str(e)}
            learned_patterns = {}
        
        # Step 3a: Learn workflow patterns from Petri nets
        workflow_patterns = None
        try:
            logger.info("Step 3a: Learning workflow patterns from Petri nets...")
            workflow_learner = WorkflowPatternLearner()
            
            # Query Petri nets from catalog via Graph service (Phase 1: optimized Neo4j access)
            if self.graph_client or self.extract_service_url:
                try:
                    # Phase 1: Use graph service client for optimized Neo4j queries
                    if self.graph_client:
                        # Query for Petri nets using graph service client
                        petri_nets_query = """
                        MATCH (n)
                        WHERE n.type = 'petri_net'
                        RETURN n.id as id, n.label as label, n.properties_json as properties
                        """
                        petri_nets_result = self.graph_client.query_neo4j(petri_nets_query)
                    else:
                        # Fallback to extract client
                        from .extract_client import ExtractServiceClient
                        extract_client = ExtractServiceClient(extract_service_url=self.extract_service_url)
                        petri_nets_query = """
                        MATCH (n)
                        WHERE n.type = 'petri_net'
                        RETURN n.id as id, n.label as label, n.properties_json as properties
                        """
                        petri_nets_result = extract_client.query_knowledge_graph(petri_nets_query)
                    
                    if petri_nets_result and petri_nets_result.get("data"):
                        # Learn from first Petri net (can be extended to learn from all)
                        petri_net_data = petri_nets_result["data"][0]
                        if petri_net_data:
                            # Get full Petri net from catalog
                            # For now, we'll use the properties from the knowledge graph
                            # In production, would fetch from catalog
                            workflow_patterns = workflow_learner.learn_from_petri_net({
                                "id": petri_net_data.get("id", "unknown"),
                                "transitions": [],
                                "arcs": [],
                                "places": [],
                            })
                            results["steps"]["workflow_patterns"] = {
                                "status": "success",
                                "patterns": len(workflow_patterns.get("workflow_patterns", {})),
                                "job_dependencies": len(workflow_patterns.get("job_dependencies", {})),
                            }
                            logger.info(f"✅ Learned workflow patterns: {results['steps']['workflow_patterns']['patterns']} patterns")
                        else:
                            results["steps"]["workflow_patterns"] = {"status": "skipped", "reason": "No Petri nets found"}
                    else:
                        results["steps"]["workflow_patterns"] = {"status": "skipped", "reason": "No Petri nets in knowledge graph"}
                except Exception as e:
                    logger.warning(f"⚠️  Workflow pattern learning failed (continuing): {e}")
                    results["steps"]["workflow_patterns"] = {"status": "failed", "error": str(e)}
            else:
                results["steps"]["workflow_patterns"] = {"status": "skipped", "reason": "Extract service not configured"}
        except Exception as e:
            logger.warning(f"⚠️  Workflow pattern learning failed: {e}")
            results["steps"]["workflow_patterns"] = {"status": "failed", "error": str(e)}
        
        # Step 3b: Analyze temporal patterns from change history
        temporal_patterns = None
        if enable_temporal_analysis:
            logger.info("Step 3b: Analyzing temporal patterns from all sources (Extract, Glean, Postgres, Neo4j)...")
            try:
                # Load json_with_changes.json if available
                json_with_changes = None
                if json_tables:
                    for json_table in json_tables:
                        if "json_with_changes" in json_table.lower() or "changes" in json_table.lower():
                            try:
                                import json as json_module
                                with open(json_table, 'r') as f:
                                    json_with_changes = json_module.load(f)
                                logger.info(f"Loaded change history from {json_table}")
                                break
                            except Exception as e:
                                logger.warning(f"Failed to load {json_table}: {e}")
                
                # Get Glean metrics if available
                glean_metrics = glean_data.get("metrics", {}) if glean_data else None
                
                # Phase 1: Use graph service client for Neo4j queries (optimized)
                # Create adapter for temporal learner (it expects extract_client interface)
                extract_client = None
                if self.graph_client:
                    # Use graph client wrapped in extract_client-compatible interface
                    # TemporalPatternLearner uses extract_client.query_knowledge_graph()
                    # We'll pass graph_client which has query_neo4j() with similar interface
                    try:
                        # Create adapter that wraps graph_client to match extract_client interface
                        class GraphClientAdapter:
                            def __init__(self, graph_client):
                                self.graph_client = graph_client
                            
                            def query_knowledge_graph(self, query, params=None):
                                result = self.graph_client.query_neo4j(query, params or {})
                                # Convert to extract_client format
                                return {
                                    "columns": result.get("columns", []),
                                    "data": result.get("data", [])
                                }
                        
                        extract_client = GraphClientAdapter(self.graph_client)
                        logger.info("Using graph service client for temporal pattern learning")
                    except Exception as e:
                        logger.warning(f"Failed to create graph client adapter: {e}")
                        extract_client = None
                
                # Fallback to extract client if graph client not available
                if not extract_client and self.extract_service_url:
                    try:
                        from .extract_client import ExtractServiceClient
                        extract_client = ExtractServiceClient(extract_service_url=self.extract_service_url)
                    except Exception as e:
                        logger.warning(f"Failed to create Extract client: {e}")
                
                # Learn temporal patterns (with graph client or extract client for Neo4j queries)
                temporal_learner = TemporalPatternLearner(
                    extract_client=extract_client,
                    glean_client=self.glean_client,
                    postgres_dsn=os.getenv("POSTGRES_CATALOG_DSN")
                )
                temporal_patterns = temporal_learner.learn_temporal_patterns(
                    json_with_changes=json_with_changes,
                    glean_metrics=glean_metrics,
                    project_id=project_id,
                    system_id=system_id
                )
                
                results["steps"]["temporal_analysis"] = {
                    "status": "success",
                    "evolution_patterns": bool(temporal_patterns.get("evolution_patterns")),
                    "temporal_metrics": bool(temporal_patterns.get("temporal_metrics")),
                    "combined_insights": bool(temporal_patterns.get("combined_insights")),
                }
                logger.info("✅ Temporal pattern analysis completed")
            except Exception as e:
                logger.warning(f"⚠️  Temporal analysis failed (continuing without temporal patterns): {e}")
                results["steps"]["temporal_analysis"] = {"status": "failed", "error": str(e)}
                temporal_patterns = None
        else:
            results["steps"]["temporal_analysis"] = {"status": "skipped"}
        
        # Step 3c: Generate GNN embeddings (Priority 1: Integration)
        # Phase 6: Parallel GNN processing for better performance
        gnn_embeddings = None
        gnn_classifications = None
        gnn_link_predictions = None
        if self.enable_gnn and self.gnn_embedder:
            logger.info("Step 3c: Generating GNN embeddings and insights (parallel processing)...")
            try:
                graph_nodes = graph_data.get("nodes", [])
                graph_edges = graph_data.get("edges", [])
                
                # Try cache first
                gnn_embeddings = None
                cache_config = {
                    "embedding_dim": getattr(self.gnn_embedder, "embedding_dim", None),
                    "hidden_dim": getattr(self.gnn_embedder, "hidden_dim", None),
                    "num_layers": getattr(self.gnn_embedder, "num_layers", None)
                }
                if self.embedding_cache and self.enable_gnn_cache:
                    cached = self.embedding_cache.get(graph_nodes, graph_edges, cache_config)
                    if cached is not None:
                        gnn_embeddings = cached
                        logger.info("Using cached GNN embeddings")
                
                # Phase 6: Parallel GNN processing using threading
                import concurrent.futures
                enable_parallel = os.getenv("ENABLE_PARALLEL_GNN", "true").lower() == "true"
                
                if gnn_embeddings is None or enable_parallel:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        # Submit parallel tasks
                        futures = {}
                        
                        # Task 1: Generate embeddings
                        if gnn_embeddings is None:
                            futures["embeddings"] = executor.submit(
                                self._generate_gnn_embeddings_parallel,
                                graph_nodes, graph_edges, cache_config
                            )
                        
                        # Task 2: Classify nodes (if enabled)
                        if self.enable_gnn_classification and self.gnn_classifier:
                            futures["classifications"] = executor.submit(
                                self._classify_nodes_parallel,
                                graph_nodes, graph_edges
                            )
                        
                        # Task 3: Predict links (if enabled)
                        if self.enable_gnn_link_prediction and self.gnn_link_predictor:
                            futures["link_predictions"] = executor.submit(
                                self._predict_links_parallel,
                                graph_nodes, graph_edges
                            )
                        
                        # Collect results
                        if "embeddings" in futures:
                            gnn_embeddings = futures["embeddings"].result()
                        
                        if "classifications" in futures:
                            gnn_classifications = futures["classifications"].result()
                        
                        if "link_predictions" in futures:
                            gnn_link_predictions = futures["link_predictions"].result()
                        
                        logger.info("✅ Parallel GNN processing completed")
                else:
                    # Sequential processing if parallel disabled or embeddings cached
                    if gnn_embeddings is None:
                        gnn_embeddings = self._generate_gnn_embeddings_parallel(graph_nodes, graph_edges, cache_config)
                
                # Process results from parallel or sequential execution
                # Handle classification results (only if not already processed in parallel)
                if self.enable_gnn_classification and self.gnn_classifier and gnn_classifications is None:
                    try:
                        # Sequential fallback if parallel didn't run
                        gnn_classifications = self._classify_nodes_parallel(graph_nodes, graph_edges)
                        if gnn_classifications and "error" not in gnn_classifications:
                            results["steps"]["gnn_classification"] = {
                                "status": "success",
                                "num_classified": len(gnn_classifications.get("classifications", [])),
                                "num_classes": len(gnn_classifications.get("class_mapping", {}))
                            }
                            logger.info(f"✅ Classified {len(gnn_classifications.get('classifications', []))} nodes")
                        else:
                            results["steps"]["gnn_classification"] = {
                                "status": "skipped",
                                "reason": gnn_classifications.get("error", "model not trained")
                            }
                            logger.info("⚠️  Node classification skipped (model not trained)")
                            # Auto-train fallback if enabled
                            try:
                                if (
                                    self.enable_gnn_training
                                    and self.gnn_trainer is not None
                                    and os.getenv("GNN_AUTO_TRAIN_ON_MISSING_MODEL", "true").lower() == "true"
                                ):
                                    err_msg = str(gnn_classifications.get("error", "")).lower()
                                    if "not trained" in err_msg:
                                        logger.info("Auto-training node classifier (fallback)...")
                                        at_epochs = int(os.getenv("GNN_AUTO_TRAIN_EPOCHS", "10"))
                                        at_lr = float(os.getenv("GNN_AUTO_TRAIN_LR", "0.01"))
                                        cls_train_result = self.gnn_trainer.train_node_classifier(
                                            graph_nodes, graph_edges, epochs=at_epochs, lr=at_lr, save_model=True
                                        )
                                        # Load and retry classification
                                        model_path = cls_train_result.get("model_path")
                                        if model_path:
                                            try:
                                                self.gnn_classifier.load_model(model_path)
                                            except Exception as e_load:
                                                logger.warning(f"Failed to load auto-trained classifier: {e_load}")
                                        retry_cls = self.gnn_classifier.classify_nodes(graph_nodes, graph_edges)
                                        if "error" not in retry_cls:
                                            gnn_classifications = retry_cls
                                            results["steps"]["gnn_classification"] = {
                                                "status": "success",
                                                "num_classified": len(retry_cls.get("classifications", [])),
                                                "num_classes": len(retry_cls.get("class_mapping", {})),
                                                "auto_trained": True
                                            }
                                            results["steps"]["gnn_training_fallback"] = results["steps"].get("gnn_training_fallback", {})
                                            results["steps"]["gnn_training_fallback"]["classifier"] = cls_train_result
                                        else:
                                            gnn_classifications = None
                                    else:
                                        gnn_classifications = None
                                else:
                                    gnn_classifications = None
                            except Exception as e_fallback:
                                logger.warning(f"Auto-train fallback for classifier failed: {e_fallback}")
                                gnn_classifications = None
                    except Exception as e:
                        logger.warning(f"⚠️  Node classification failed: {e}")
                        results["steps"]["gnn_classification"] = {"status": "failed", "error": str(e)}
                        gnn_classifications = None
                else:
                    results["steps"]["gnn_classification"] = {"status": "skipped", "reason": "disabled"}
                
                # Link prediction if enabled (only if not already processed in parallel)
                if self.enable_gnn_link_prediction and self.gnn_link_predictor and gnn_link_predictions is None:
                    try:
                        # Sequential fallback if parallel didn't run
                        gnn_link_predictions = self._predict_links_parallel(graph_nodes, graph_edges)
                        if gnn_link_predictions and "error" not in gnn_link_predictions:
                            results["steps"]["gnn_link_prediction"] = {
                                "status": "success",
                                "num_predictions": len(gnn_link_predictions.get("predictions", [])),
                                "num_candidates": gnn_link_predictions.get("num_candidates", 0)
                            }
                            logger.info(f"✅ Predicted {len(gnn_link_predictions.get('predictions', []))} potential links")
                        else:
                            results["steps"]["gnn_link_prediction"] = {
                                "status": "skipped",
                                "reason": gnn_link_predictions.get("error", "model not trained")
                            }
                            logger.info("⚠️  Link prediction skipped (model not trained)")
                            # Auto-train fallback if enabled
                            try:
                                if (
                                    self.enable_gnn_training
                                    and self.gnn_trainer is not None
                                    and os.getenv("GNN_AUTO_TRAIN_ON_MISSING_MODEL", "true").lower() == "true"
                                ):
                                    err_msg = str(gnn_link_predictions.get("error", "")).lower()
                                    if "not trained" in err_msg:
                                        logger.info("Auto-training link predictor (fallback)...")
                                        at_epochs = int(os.getenv("GNN_AUTO_TRAIN_EPOCHS", "10"))
                                        at_lr = float(os.getenv("GNN_AUTO_TRAIN_LR", "0.01"))
                                        at_neg = int(os.getenv("GNN_AUTO_TRAIN_NEG_SAMPLES", "1"))
                                        pred_train_result = self.gnn_trainer.train_link_predictor(
                                            graph_nodes, graph_edges, epochs=at_epochs, lr=at_lr
                                        )
                                        # Load and retry prediction
                                        model_path = pred_train_result.get("model_path")
                                        if model_path:
                                            try:
                                                self.gnn_link_predictor.load_model(model_path)
                                            except Exception as e_load:
                                                logger.warning(f"Failed to load auto-trained link predictor: {e_load}")
                                        retry_pred = self.gnn_link_predictor.predict_links(
                                            graph_nodes, graph_edges,
                                            top_k=int(os.getenv("GNN_LINK_PREDICTION_TOP_K", "10"))
                                        )
                                        if "error" not in retry_pred:
                                            gnn_link_predictions = retry_pred
                                            results["steps"]["gnn_link_prediction"] = {
                                                "status": "success",
                                                "num_predictions": len(retry_pred.get("predictions", [])),
                                                "num_candidates": retry_pred.get("num_candidates", 0),
                                                "auto_trained": True
                                            }
                                            results["steps"]["gnn_training_fallback"] = results["steps"].get("gnn_training_fallback", {})
                                            results["steps"]["gnn_training_fallback"]["link_predictor"] = pred_train_result
                                        else:
                                            gnn_link_predictions = None
                                    else:
                                        gnn_link_predictions = None
                                else:
                                    gnn_link_predictions = None
                            except Exception as e_fallback:
                                logger.warning(f"Auto-train fallback for link predictor failed: {e_fallback}")
                                gnn_link_predictions = None
                    except Exception as e:
                        logger.warning(f"⚠️  Link prediction failed: {e}")
                        results["steps"]["gnn_link_prediction"] = {"status": "failed", "error": str(e)}
                        gnn_link_predictions = None
                else:
                    results["steps"]["gnn_link_prediction"] = {"status": "skipped", "reason": "disabled"}
                
                if "error" not in gnn_embeddings:
                    results["steps"]["gnn_embeddings"] = {
                        "status": "success",
                        "embedding_dim": gnn_embeddings.get("embedding_dim", 128),
                        "num_nodes": gnn_embeddings.get("num_nodes", 0),
                        "num_edges": gnn_embeddings.get("num_edges", 0)
                    }
                    logger.info(f"✅ Generated GNN embeddings (dim: {gnn_embeddings.get('embedding_dim', 'unknown')})")
                else:
                    results["steps"]["gnn_embeddings"] = {
                        "status": "failed",
                        "error": gnn_embeddings.get("error", "unknown error")
                    }
                    logger.warning(f"⚠️  GNN embedding generation failed: {gnn_embeddings.get('error')}")
                    gnn_embeddings = None
            except Exception as e:
                logger.warning(f"⚠️  GNN processing failed (continuing without GNN features): {e}")
                results["steps"]["gnn_embeddings"] = {"status": "failed", "error": str(e)}
                gnn_embeddings = None
        else:
            results["steps"]["gnn_embeddings"] = {"status": "skipped", "reason": "disabled or not available"}
            if "gnn_classification" not in results["steps"]:
                results["steps"]["gnn_classification"] = {"status": "skipped", "reason": "disabled or not available"}
            if "gnn_link_prediction" not in results["steps"]:
                results["steps"]["gnn_link_prediction"] = {"status": "skipped", "reason": "disabled or not available"}
        
        # Step 3d: Get semantic embeddings for training features
        semantic_embeddings = None
        if os.getenv("USE_SAP_RPT_EMBEDDINGS", "false").lower() == "true":
            logger.info("Step 3d: Retrieving semantic embeddings...")
            try:
                semantic_embeddings = self._get_semantic_embeddings_for_training(
                    graph_data=graph_data,
                    extract_service_url=self.extract_service_url
                )
                logger.info(f"✅ Retrieved semantic embeddings for {len(semantic_embeddings) if semantic_embeddings else 0} artifacts")
            except Exception as e:
                logger.warning(f"⚠️ Failed to retrieve semantic embeddings: {e}")
                semantic_embeddings = None
        
        # Step 4: Generate training features
        logger.info("Step 4: Generating training features...")
        try:
            features = self._generate_training_features(
                graph_data, glean_data, learned_patterns, temporal_patterns, semantic_embeddings,
                gnn_embeddings=gnn_embeddings,
                gnn_classifications=gnn_classifications,
                gnn_link_predictions=gnn_link_predictions
            )
            # Apply domain-specific filtering with differential privacy
            if self.domain_filter and self.enable_domain_filtering:
                logger.info("Applying domain-specific filtering with differential privacy...")
                domain_features = features.get("features", [])
                
                # Filter features by domain (auto-detect domain from graph)
                filtered_features = self.domain_filter.filter_features_by_domain(
                    domain_features,
                    domain_id=None  # Auto-detect
                )
                
                if filtered_features:
                    features["features"] = filtered_features
                    features["domain_filtered"] = True
                    features["privacy_applied"] = True
                    
                    # Add privacy stats
                    privacy_stats = self.domain_filter.get_privacy_stats()
                    features["privacy_stats"] = privacy_stats
                    
                    logger.info(
                        f"✅ Applied domain filtering: {len(filtered_features)}/{len(domain_features)} "
                        f"features (privacy: ε={self.domain_filter.privacy_config.epsilon}, "
                        f"δ={self.domain_filter.privacy_config.delta})"
                    )
                else:
                    logger.warning("⚠️  No features matched domain filter, using all features")
                    features["domain_filtered"] = False
                    features["privacy_applied"] = False
            else:
                features["domain_filtered"] = False
                features["privacy_applied"] = False

            # Phase 9.1: Assess training data quality
            if self.auto_tuner is not None:
                try:
                    samples = features.get("features", [])
                    pattern_summary = (learned_patterns or {}).get("summary", {})
                    num_samples = len(samples)
                    num_features = len(samples[0]) if num_samples > 0 and isinstance(samples[0], (list, tuple)) else 0
                    pattern_coverage = float(pattern_summary.get("unique_column_types", 0)) / max(
                        float(pattern_summary.get("total_column_types", 1) or 1), 1.0
                    )
                    training_data_stats = {
                        "num_samples": num_samples,
                        "num_features": num_features,
                        "pattern_coverage": pattern_coverage,
                    }
                    quality_assessment = self.auto_tuner.assess_training_data_quality(training_data_stats)
                    features["quality_assessment"] = quality_assessment
                    results["steps"]["data_quality"] = {
                        "status": "success",
                        "quality_score": quality_assessment.get("quality_score", 0.0),
                        "passed": quality_assessment.get("passed", False),
                    }
                    logger.info(
                        f"✅ Data quality assessment: score="
                        f"{quality_assessment.get('quality_score', 0.0):.2f}"
                    )
                except Exception as e:
                    logger.warning(f"⚠️  Data quality assessment failed: {e}")
            
            # Priority 2: Evaluate GNN models if available and persist results (Phase 3)
            gnn_evaluation_results = None
            if self.enable_gnn and HAS_GNN_TRAINING and GNNEvaluator is not None:
                try:
                    gnn_evaluation_results = self._evaluate_gnn_models(
                        graph_data, gnn_embeddings, gnn_classifications, gnn_link_predictions
                    )
                    if gnn_evaluation_results:
                        results["steps"]["gnn_evaluation"] = gnn_evaluation_results
                        logger.info("✅ GNN model evaluation completed")
                        # Phase 3: persist evaluation results to disk
                        try:
                            eval_dir = os.path.join(self.output_dir, "gnn_eval")
                            os.makedirs(eval_dir, exist_ok=True)
                            eval_file = os.path.join(eval_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                            with open(eval_file, "w") as f:
                                json.dump(gnn_evaluation_results, f, indent=2)
                            # Also add a latest symlink or copy
                            latest_file = os.path.join(eval_dir, "latest.json")
                            try:
                                # On systems without symlink permissions, fall back to copy
                                if os.path.islink(latest_file) or os.path.exists(latest_file):
                                    try:
                                        os.remove(latest_file)
                                    except Exception:
                                        pass
                                os.symlink(os.path.basename(eval_file), latest_file)
                            except Exception:
                                try:
                                    with open(latest_file, "w") as f2:
                                        json.dump(gnn_evaluation_results, f2, indent=2)
                                except Exception:
                                    pass
                            results["steps"]["gnn_evaluation"]["persisted_to"] = eval_file
                        except Exception as e_persist:
                            logger.warning(f"Failed to persist GNN eval results: {e_persist}")
                except Exception as e:
                    logger.warning(f"⚠️  GNN evaluation failed: {e}")
            
            results["steps"]["features"] = {
                "status": "success",
                "feature_count": len(features.get("features", [])),
                "domain_filtered": features.get("domain_filtered", False),
                "privacy_applied": features.get("privacy_applied", False),
            }
            logger.info(f"✅ Generated {results['steps']['features']['feature_count']} training features")
        except Exception as e:
            logger.error(f"❌ Feature generation failed: {e}")
            results["steps"]["features"] = {"status": "failed", "error": str(e)}
            return results
        
        # Step 5: Prepare training dataset
        dataset_info = None
        logger.info("Step 5: Preparing training dataset...")
        try:
            dataset_info = self._prepare_training_dataset(features, output_dir=self.output_dir)
            results["steps"]["dataset"] = {
                "status": "success",
                "dataset_files": dataset_info.get("files", []),
            }
            logger.info(f"✅ Prepared training dataset with {len(dataset_info.get('files', []))} files")
        except Exception as e:
            logger.error(f"❌ Dataset preparation failed: {e}")
            results["steps"]["dataset"] = {"status": "failed", "error": str(e)}
            return results

        # Step 6: Run digital twin simulation (optional)
        if enable_digital_twin and self.digital_twin.is_enabled:
            logger.info("Step 6: Running digital twin simulation...")
            try:
                simulation_result = self.digital_twin.simulate(
                    pipeline_results=results,
                    dataset_info=dataset_info,
                    signavio_metadata=signavio_summary if isinstance(signavio_summary, dict) else None,
                )
                results["steps"]["digital_twin"] = simulation_result
                if simulation_result.get("status") == "success":
                    logger.info(
                        "✅ Digital twin simulation completed (%s mode)",
                        simulation_result.get("mode", "local"),
                    )
                else:
                    logger.warning(
                        "⚠️ Digital twin simulation returned status=%s",
                        simulation_result.get("status"),
                    )
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(f"⚠️ Digital twin simulation failure: {e}")
                results["steps"]["digital_twin"] = {"status": "failed", "error": str(e)}
        else:
            results["steps"]["digital_twin"] = {"status": "skipped"}

        # Step 7: Domain-specific model training (if enabled)
        if os.getenv("ENABLE_DOMAIN_TRAINING", "false").lower() == "true":
            logger.info("Step 7: Training domain-specific models...")
            try:
                # Detect domain from extracted data
                domain_id = self._detect_domain_from_results(results)
                
                if domain_id:
                    # Prepare training data path
                    dataset_files = (dataset_info or {}).get("files", []) or results.get("steps", {}).get("dataset", {}).get("dataset_files", [])
                    
                    if dataset_files:
                        training_data_path = dataset_files[0]  # Use first dataset file
                        
                        # Train domain-specific model
                        training_result = self.domain_trainer.train_domain_model(
                            domain_id=domain_id,
                            training_data_path=training_data_path,
                            fine_tune=True
                        )
                        
                        results["steps"]["domain_training"] = {
                            "status": "success",
                            "domain_id": domain_id,
                            "training_run_id": training_result.get("training_run_id"),
                            "should_deploy": training_result.get("should_deploy", False),
                            "deployment": training_result.get("deployment"),
                        }
                        
                        if training_result.get("should_deploy"):
                            logger.info(f"✅ Domain model trained and deployed for {domain_id}")
                        else:
                            logger.info(f"✅ Domain model trained for {domain_id} (deployment threshold not met)")
                    else:
                        results["steps"]["domain_training"] = {
                            "status": "skipped",
                            "reason": "No training dataset available"
                        }
                else:
                    results["steps"]["domain_training"] = {
                        "status": "skipped",
                        "reason": "Could not detect domain"
                    }
            except Exception as e:
                logger.error(f"❌ Domain training failed: {e}")
                results["steps"]["domain_training"] = {"status": "failed", "error": str(e)}
        else:
            results["steps"]["domain_training"] = {"status": "skipped"}
        
        # Step 8: Collect domain metrics
        logger.info("Step 8: Collecting domain performance metrics...")
        try:
            domain_id = self._detect_domain_from_results(results)
            if domain_id:
                metrics = self.metrics_collector.collect_domain_metrics(
                    domain_id=domain_id,
                    time_window_days=30
                )
                results["domain_metrics"] = metrics
                results["steps"]["metrics_collection"] = {
                    "status": "success",
                    "domain_id": domain_id,
                }
                logger.info(f"✅ Collected metrics for domain {domain_id}")
            else:
                results["steps"]["metrics_collection"] = {
                    "status": "skipped",
                    "reason": "Could not detect domain"
                }
        except Exception as e:
            logger.warning(f"⚠️  Metrics collection failed: {e}")
            results["steps"]["metrics_collection"] = {"status": "failed", "error": str(e)}
        
        # Step 9: Check for rollback conditions (if domain training was performed)
        if results.get("steps", {}).get("domain_training", {}).get("status") == "success":
            logger.info("Step 9: Checking for rollback conditions...")
            try:
                domain_id = self._detect_domain_from_results(results)
                if domain_id:
                    # Get current metrics from training
                    training_result = results["steps"]["domain_training"]
                    if training_result.get("deployment"):
                        deployment_metrics = training_result["deployment"].get("metrics", {})
                        
                        # Check rollback
                        rollback_result = self.rollback_manager.check_and_rollback(
                            domain_id=domain_id,
                            current_metrics=deployment_metrics
                        )
                        
                        results["steps"]["rollback_check"] = {
                            "status": "success",
                            "rollback_triggered": rollback_result.get("rollback_triggered", False),
                            "reason": rollback_result.get("reason"),
                        }
                        
                        if rollback_result.get("rollback_triggered"):
                            logger.warning(f"⚠️  Rollback triggered for {domain_id}: {rollback_result.get('reason')}")
                        else:
                            logger.info(f"✅ No rollback needed for {domain_id}")
                    else:
                        results["steps"]["rollback_check"] = {
                            "status": "skipped",
                            "reason": "No deployment performed"
                        }
                else:
                    results["steps"]["rollback_check"] = {
                        "status": "skipped",
                        "reason": "Could not detect domain"
                    }
            except Exception as e:
                logger.warning(f"⚠️  Rollback check failed: {e}")
                results["steps"]["rollback_check"] = {"status": "failed", "error": str(e)}
        else:
            results["steps"]["rollback_check"] = {"status": "skipped"}
        
        results["pipeline_completed_at"] = datetime.now().isoformat()
        results["status"] = "success"

        self.langsmith_tracer.record_run(
            project_id=project_id,
            system_id=system_id,
            results=results,
            dataset_info=dataset_info,
        )

        logger.info("✅ Training pipeline completed successfully")
        
        return results
    
    def _detect_domain_from_results(self, results: Dict[str, Any]) -> Optional[str]:
        """Detect domain ID from pipeline results."""
        # Try to get domain from filtered features
        features = results.get("steps", {}).get("features", {})
        if features.get("domain_filtered"):
            # Domain was detected during filtering
            # Try to extract from graph data
            extract_step = results.get("steps", {}).get("extract", {})
            if extract_step.get("status") == "success":
                # Would need to query graph data for domain
                # For now, return None and let user specify
                pass
        
        # Return None if can't detect (user can specify manually)
        return None
    
    def _extract_knowledge_graph(
        self,
        project_id: str,
        system_id: Optional[str],
        json_tables: list,
        hive_ddls: list,
        control_m_files: list,
        signavio_files: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Extract knowledge graph from source data via Extract service or Graph service.
        
        Phase 1: Uses Graph service client for optimized Neo4j access when available,
        falls back to Extract service for backward compatibility.
        
        Improvement 4: Uses unified data access layer when available for consistent data access.
        Improvement 6: Uses cache for graph data retrieval.
        
        Note: Knowledge graph extraction still uses Extract service endpoint as it processes
        source files. Graph service client is used for subsequent Neo4j queries.
        """
        import httpx
        
        # Improvement 4 & 6: Try unified data access with cache first
        if self.unified_data_access:
            try:
                cached_graph = self.unified_data_access.cache_manager.get_cached_graph_data(project_id, system_id)
                if cached_graph:
                    logger.info("Using cached graph data from unified data access layer")
                    return cached_graph
                
                # Get from unified data access
                graph_data = self.unified_data_access.get_graph_data(project_id, system_id, use_cache=True)
                if graph_data and len(graph_data.nodes) > 0:
                    # Cache the result
                    self.unified_data_access.cache_manager.cache_graph_data(project_id, system_id, graph_data)
                    return {
                        "nodes": [n.dict() if hasattr(n, 'dict') else n for n in graph_data.nodes],
                        "edges": [e.dict() if hasattr(e, 'dict') else e for e in graph_data.edges]
                    }
            except Exception as e:
                logger.warning(f"Unified data access failed, falling back to extract service: {e}")
        
        payload = {
            "json_tables": json_tables,
            "hive_ddls": hive_ddls,
            "sql_queries": [],
            "control_m_files": control_m_files,
            "signavio_files": signavio_files or [],
            "project_id": project_id,
            "system_id": system_id,
        }
        
        # Knowledge graph extraction uses Extract service (processes source files)
        # Graph service client will be used for subsequent Neo4j queries
        client = httpx.Client(timeout=300.0)
        try:
            response = client.post(
                f"{self.extract_service_url}/knowledge-graph",
                json=payload,
            )
            response.raise_for_status()
            graph_data = response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning(
                "Extract service unavailable, continuing with Signavio-only data: %s",
                exc,
            )
            return {
                "nodes": [],
                "edges": [],
                "signavio": {
                    "process_count": 0,
                    "source_files": len(signavio_files or []),
                    "error": str(exc),
                },
                "service_unavailable": True,
            }

        # Apply domain-specific filtering with differential privacy if enabled
        if self.domain_filter and self.enable_domain_filtering:
            logger.info("Applying domain filtering to extracted knowledge graph...")
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            
            # Filter by domain (auto-detect)
            filtered_nodes, filtered_edges = self.domain_filter.filter_by_domain(
                nodes, edges, domain_id=None
            )
            
            if filtered_nodes or filtered_edges:
                graph_data["nodes"] = filtered_nodes
                graph_data["edges"] = filtered_edges
                graph_data["domain_filtered"] = True
                graph_data["privacy_applied"] = True
                logger.info(
                    f"✅ Filtered graph: {len(filtered_nodes)} nodes, {len(filtered_edges)} edges "
                    f"(privacy: ε={self.domain_filter.privacy_config.epsilon})"
                )
            else:
                graph_data["domain_filtered"] = False
                graph_data["privacy_applied"] = False
        else:
            graph_data["domain_filtered"] = False
            graph_data["privacy_applied"] = False
        
        return graph_data
    
    def _generate_training_features(
        self,
        graph_data: Dict[str, Any],
        glean_data: Optional[Dict[str, Any]],
        learned_patterns: Optional[Dict[str, Any]] = None,
        temporal_patterns: Optional[Dict[str, Any]] = None,
        semantic_embeddings: Optional[Dict[str, Any]] = None,
        gnn_embeddings: Optional[Dict[str, Any]] = None,
        gnn_classifications: Optional[Dict[str, Any]] = None,
        gnn_link_predictions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate training features from graph data and Glean patterns.
        
        Priority 1: Now includes GNN embeddings to replace/enhance manual features.
        """
        features = []
        
        # Priority 1: Add GNN embeddings as primary features (replace manual features if available)
        if gnn_embeddings and "error" not in gnn_embeddings:
            # Graph-level embedding
            if "graph_embedding" in gnn_embeddings:
                features.append({
                    "type": "gnn_graph_embedding",
                    "data": gnn_embeddings["graph_embedding"],
                    "embedding_dim": gnn_embeddings.get("embedding_dim", 128),
                    "source": "gnn_embedder"
                })
            
            # Node-level embeddings
            if "node_embeddings" in gnn_embeddings:
                features.append({
                    "type": "gnn_node_embeddings",
                    "data": gnn_embeddings["node_embeddings"],
                    "embedding_dim": gnn_embeddings.get("embedding_dim", 128),
                    "num_nodes": gnn_embeddings.get("num_nodes", 0),
                    "source": "gnn_embedder"
                })
            
            logger.info("✅ Using GNN embeddings as primary features")
        else:
            # Fallback to manual features if GNN not available
            logger.info("Using manual feature engineering (GNN embeddings not available)")
            
            # Extract features from current knowledge graph (fallback)
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            
            # Feature: Node type distribution
            node_types = {}
            for node in nodes:
                node_type = node.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Feature: Edge label distribution
            edge_labels = {}
            for edge in edges:
                label = edge.get("label", "unknown")
                edge_labels[label] = edge_labels.get(label, 0) + 1
            
            # Feature: Information theory metrics (if available)
            metrics = graph_data.get("metrics", {})
            
            features.append({
                "type": "node_type_distribution",
                "data": node_types,
            })
            features.append({
                "type": "edge_label_distribution",
                "data": edge_labels,
            })
            
            if metrics:
                features.append({
                    "type": "information_theory_metrics",
                    "data": metrics,
                })
        
        # Add GNN classifications if available
        if gnn_classifications and "error" not in gnn_classifications:
            features.append({
                "type": "gnn_node_classifications",
                "data": gnn_classifications.get("classifications", []),
                "class_mapping": gnn_classifications.get("class_mapping", {}),
                "source": "gnn_classifier"
            })
        
        # Add GNN link predictions if available
        if gnn_link_predictions and "error" not in gnn_link_predictions:
            features.append({
                "type": "gnn_link_predictions",
                "data": gnn_link_predictions.get("predictions", []),
                "num_candidates": gnn_link_predictions.get("num_candidates", 0),
                "source": "gnn_link_predictor"
            })
        
        # Add historical patterns from Glean if available
        if glean_data:
            glean_metrics = glean_data.get("metrics", {})
            if glean_metrics:
                features.append({
                    "type": "historical_metrics_trend",
                    "data": glean_metrics,
                })
            
            column_patterns = glean_data.get("column_patterns", {})
            if column_patterns:
                features.append({
                    "type": "column_type_patterns",
                    "data": column_patterns,
                })
        
        # Add learned patterns if available
        if learned_patterns:
            features.append({
                "type": "learned_column_patterns",
                "data": learned_patterns.get("column_patterns", {}),
            })
            features.append({
                "type": "learned_relationship_patterns",
                "data": learned_patterns.get("relationship_patterns", {}),
            })
            features.append({
                "type": "learned_metrics_patterns",
                "data": learned_patterns.get("metrics_patterns", {}),
            })
        
        # Add temporal patterns if available
        if temporal_patterns:
            features.append({
                "type": "temporal_evolution_patterns",
                "data": temporal_patterns.get("evolution_patterns", {}),
            })
            features.append({
                "type": "temporal_metrics_patterns",
                "data": temporal_patterns.get("temporal_metrics", {}),
            })
            features.append({
                "type": "temporal_combined_insights",
                "data": temporal_patterns.get("combined_insights", {}),
            })
        
        # Get node/edge counts
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        return {
            "features": features,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "has_historical_data": glean_data is not None,
            "has_learned_patterns": learned_patterns is not None,
            "has_temporal_patterns": temporal_patterns is not None,
            "has_gnn_embeddings": gnn_embeddings is not None and "error" not in (gnn_embeddings or {}),
            "has_gnn_classifications": gnn_classifications is not None and "error" not in (gnn_classifications or {}),
            "has_gnn_link_predictions": gnn_link_predictions is not None and "error" not in (gnn_link_predictions or {}),
        }
    
    def _evaluate_gnn_models(
        self,
        graph_data: Dict[str, Any],
        gnn_embeddings: Optional[Dict[str, Any]],
        gnn_classifications: Optional[Dict[str, Any]],
        gnn_link_predictions: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate GNN models (Priority 2).
        
        Args:
            graph_data: Graph data
            gnn_embeddings: GNN embeddings results
            gnn_classifications: GNN classification results
            gnn_link_predictions: GNN link prediction results
        
        Returns:
            Dictionary with evaluation results
        """
        if not HAS_GNN_TRAINING or GNNEvaluator is None:
            return None
        
        evaluator = GNNEvaluator()
        evaluation_results = {}
        
        # Evaluate embeddings
        if gnn_embeddings and "error" not in gnn_embeddings:
            try:
                if "node_embeddings" in gnn_embeddings:
                    # Extract embeddings as numpy array
                    node_embeddings_dict = gnn_embeddings["node_embeddings"]
                    if isinstance(node_embeddings_dict, dict):
                        embeddings_list = list(node_embeddings_dict.values())
                        if embeddings_list:
                            import numpy as np
                            embeddings_array = np.array(embeddings_list)
                            
                            # Extract labels from nodes for evaluation
                            nodes = graph_data.get("nodes", [])
                            labels = [node.get("type", "unknown") for node in nodes[:len(embeddings_array)]]
                            
                            if len(labels) == len(embeddings_array):
                                embedding_metrics = evaluator.evaluate_embeddings(
                                    embeddings_array,
                                    labels=labels
                                )
                                evaluation_results["embedding_quality"] = embedding_metrics
            except Exception as e:
                logger.warning(f"Failed to evaluate embeddings: {e}")
        
        # Evaluate classifications
        if gnn_classifications and "error" not in gnn_classifications:
            try:
                # Extract true labels from nodes
                nodes = graph_data.get("nodes", [])
                classifications = gnn_classifications.get("classifications", [])
                
                if classifications:
                    y_true = []
                    y_pred = []
                    for classification in classifications:
                        node_id = classification["node_id"]
                        # Find corresponding node
                        for node in nodes:
                            if node.get("id", "") == node_id:
                                y_true.append(node.get("type", "unknown"))
                                y_pred.append(classification["predicted_class"])
                                break
                    
                    if y_true and y_pred:
                        class_mapping = gnn_classifications.get("class_mapping", {})
                        class_names = [class_mapping.get(str(i), f"class_{i}") for i in range(len(class_mapping))]
                        classification_metrics = evaluator.evaluate_classification(
                            y_true, y_pred, class_names=class_names
                        )
                        evaluation_results["classification"] = classification_metrics
            except Exception as e:
                logger.warning(f"Failed to evaluate classifications: {e}")
        
        # Evaluate link predictions (limited - need ground truth)
        if gnn_link_predictions and "error" not in gnn_link_predictions:
            try:
                predictions = gnn_link_predictions.get("predictions", [])
                if predictions:
                    # Note: Full evaluation requires ground truth links
                    # For now, just report statistics
                    probabilities = [p.get("probability", 0.0) for p in predictions]
                    evaluation_results["link_prediction"] = {
                        "num_predictions": len(predictions),
                        "mean_probability": float(sum(probabilities) / len(probabilities)) if probabilities else 0.0,
                        "max_probability": float(max(probabilities)) if probabilities else 0.0,
                        "min_probability": float(min(probabilities)) if probabilities else 0.0,
                        "note": "Full evaluation requires ground truth links"
                    }
            except Exception as e:
                logger.warning(f"Failed to evaluate link predictions: {e}")
        
        return evaluation_results if evaluation_results else None
    
    def train_gnn_models(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        train_classifier: bool = True,
        train_link_predictor: bool = True,
        train_anomaly_detector: bool = False,
        epochs: int = 100,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """Train GNN models (Priority 2).
        
        Args:
            nodes: Training nodes
            edges: Training edges
            train_classifier: Whether to train node classifier
            train_link_predictor: Whether to train link predictor
            train_anomaly_detector: Whether to train anomaly detector
            epochs: Number of training epochs
            lr: Learning rate
        
        Returns:
            Dictionary with training results
        """
        if not self.enable_gnn_training or self.gnn_trainer is None:
            return {"error": "GNN training not enabled or trainer not initialized"}
        
        logger.info("Training GNN models...")
        
        results = {
            "status": "success",
            "models_trained": [],
            "training_results": {}
        }
        
        # Train node classifier
        if train_classifier:
            try:
                classifier_result = self.gnn_trainer.train_node_classifier(
                    nodes, edges,
                    epochs=epochs,
                    lr=lr
                )
                results["training_results"]["classifier"] = classifier_result
                results["models_trained"].append("classifier")
                logger.info(f"✅ Node classifier trained: accuracy={classifier_result.get('training_accuracy', 0.0):.4f}")
            except Exception as e:
                logger.error(f"❌ Node classifier training failed: {e}")
                results["training_results"]["classifier"] = {"error": str(e)}
        
        # Train link predictor
        if train_link_predictor:
            try:
                predictor_result = self.gnn_trainer.train_link_predictor(
                    nodes, edges,
                    epochs=epochs,
                    lr=lr
                )
                results["training_results"]["link_predictor"] = predictor_result
                results["models_trained"].append("link_predictor")
                logger.info(f"✅ Link predictor trained: accuracy={predictor_result.get('training_accuracy', 0.0):.4f}")
            except Exception as e:
                logger.error(f"❌ Link predictor training failed: {e}")
                results["training_results"]["link_predictor"] = {"error": str(e)}
        
        # Train anomaly detector
        if train_anomaly_detector:
            try:
                detector_result = self.gnn_trainer.train_anomaly_detector(
                    nodes, edges,
                    epochs=epochs,
                    lr=lr
                )
                results["training_results"]["anomaly_detector"] = detector_result
                results["models_trained"].append("anomaly_detector")
                logger.info(f"✅ Anomaly detector trained: loss={detector_result.get('final_loss', 0.0):.4f}")
            except Exception as e:
                logger.error(f"❌ Anomaly detector training failed: {e}")
                results["training_results"]["anomaly_detector"] = {"error": str(e)}
        
        logger.info(f"✅ GNN training complete: {len(results['models_trained'])} models trained")
        
        return results
    
    def load_trained_gnn_models(
        self,
        model_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        """Load trained GNN models.
        
        Args:
            model_paths: Dictionary mapping model type to path
        
        Returns:
            Dictionary with loaded models
        """
        if not self.enable_gnn_training or self.gnn_trainer is None:
            return {"error": "GNN training not enabled or trainer not initialized"}
        
        return self.gnn_trainer.load_trained_models(model_paths)
    
    def _prepare_training_dataset(
        self,
        features: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """Prepare training dataset files."""
        dataset_dir = os.path.join(output_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save features to JSON
        features_file = os.path.join(dataset_dir, "features.json")
        with open(features_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        # Save metadata
        metadata_file = os.path.join(dataset_dir, "metadata.json")
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "feature_count": len(features.get("features", [])),
            "node_count": features.get("node_count", 0),
            "edge_count": features.get("edge_count", 0),
            "has_historical_data": features.get("has_historical_data", False),
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "files": [features_file, metadata_file],
            "metadata": metadata,
        }
    
    def _generate_gnn_embeddings_parallel(
        self,
        graph_nodes: List[Dict[str, Any]],
        graph_edges: List[Dict[str, Any]],
        cache_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 6: Generate GNN embeddings (parallel processing helper).
        
        Args:
            graph_nodes: Graph nodes
            graph_edges: Graph edges
            cache_config: Cache configuration
        
        Returns:
            Embeddings dictionary
        """
        # Generate graph-level and node-level embeddings
        gnn_embeddings = self.gnn_embedder.generate_embeddings(
            graph_nodes,
            graph_edges,
            graph_level=True
        )

        # Optimize embedder model for inference after first init
        if (
            self.enable_gnn_inference_opt
            and HAS_GNN_PRIORITY4
            and getattr(self.gnn_embedder, "model", None) is not None
            and not self._embedder_optimized
            and self.gnn_inference_optimizer is not None
        ):
            try:
                self.gnn_embedder.model = self.gnn_inference_optimizer.optimize_for_inference(
                    self.gnn_embedder.model, None
                )
                self._embedder_optimized = True
                logger.info("Optimized GNN embedder model for inference")
            except Exception as e:
                logger.warning(f"Failed to optimize embedder model: {e}")
        
        node_embeddings = self.gnn_embedder.generate_embeddings(
            graph_nodes,
            graph_edges,
            graph_level=False
        )
        
        if "error" not in gnn_embeddings and "error" not in node_embeddings:
            gnn_embeddings["node_embeddings"] = node_embeddings.get("node_embeddings", {})
            if self.embedding_cache and self.enable_gnn_cache:
                self.embedding_cache.put(graph_nodes, graph_edges, gnn_embeddings, cache_config)
                logger.info("Cached GNN embeddings")
        
        return gnn_embeddings
    
    def _classify_nodes_parallel(
        self,
        graph_nodes: List[Dict[str, Any]],
        graph_edges: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Phase 6: Classify nodes (parallel processing helper).
        
        Args:
            graph_nodes: Graph nodes
            graph_edges: Graph edges
        
        Returns:
            Classifications dictionary or None
        """
        try:
            # Optimize classifier model for inference (once)
            if (
                self.enable_gnn_inference_opt
                and HAS_GNN_PRIORITY4
                and getattr(self.gnn_classifier, "model", None) is not None
                and not self._classifier_optimized
                and self.gnn_inference_optimizer is not None
            ):
                try:
                    self.gnn_classifier.model = self.gnn_inference_optimizer.optimize_for_inference(
                        self.gnn_classifier.model, None
                    )
                    self._classifier_optimized = True
                    logger.info("Optimized GNN classifier model for inference")
                except Exception as e:
                    logger.warning(f"Failed to optimize classifier model: {e}")
            
            return self.gnn_classifier.classify_nodes(graph_nodes, graph_edges)
        except Exception as e:
            logger.warning(f"Parallel node classification failed: {e}")
            return None
    
    def _predict_links_parallel(
        self,
        graph_nodes: List[Dict[str, Any]],
        graph_edges: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Phase 6: Predict links (parallel processing helper).
        
        Args:
            graph_nodes: Graph nodes
            graph_edges: Graph edges
        
        Returns:
            Link predictions dictionary or None
        """
        try:
            # Optimize link predictor model for inference (once)
            if (
                self.enable_gnn_inference_opt
                and HAS_GNN_PRIORITY4
                and getattr(self.gnn_link_predictor, "model", None) is not None
                and not self._link_predictor_optimized
                and self.gnn_inference_optimizer is not None
            ):
                try:
                    self.gnn_link_predictor.model = self.gnn_inference_optimizer.optimize_for_inference(
                        self.gnn_link_predictor.model, None
                    )
                    self._link_predictor_optimized = True
                    logger.info("Optimized GNN link predictor model for inference")
                except Exception as e:
                    logger.warning(f"Failed to optimize link predictor model: {e}")
            
            return self.gnn_link_predictor.predict_links(
                graph_nodes,
                graph_edges,
                top_k=int(os.getenv("GNN_LINK_PREDICTION_TOP_K", "10"))
            )
        except Exception as e:
            logger.warning(f"Parallel link prediction failed: {e}")
            return None
