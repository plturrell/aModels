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
from typing import Dict, Optional, Any
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

# Phase 9.1: Auto-tuning
try:
    from .auto_tuner import AutoTuner
    HAS_AUTO_TUNER = True
except ImportError:
    HAS_AUTO_TUNER = False
    AutoTuner = None

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline orchestrator."""
    
    def __init__(
        self,
        extract_service_url: Optional[str] = None,
        glean_db_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        enable_domain_filtering: bool = True,
        privacy_level: str = "medium"
    ):
        self.extract_service_url = extract_service_url or os.getenv("EXTRACT_SERVICE_URL", "http://localhost:19080")
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
Wait we introduced typo. Need to patch carefully. Let's reapply patch carefully. Go back. We'll revert snippet. Let's redo patch.
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
            
            # Query Petri nets from catalog via Extract service
            if self.extract_service_url:
                try:
                    from .extract_client import ExtractServiceClient
                    extract_client = ExtractServiceClient(extract_service_url=self.extract_service_url)
                    
                    # Query for Petri nets
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
                
                # Create Extract client for Neo4j queries
                extract_client = None
                if self.extract_service_url:
                    try:
                        from .extract_client import ExtractServiceClient
                        extract_client = ExtractServiceClient(extract_service_url=self.extract_service_url)
                    except Exception as e:
                        logger.warning(f"Failed to create Extract client: {e}")
                
                # Learn temporal patterns (with Extract client for Neo4j queries)
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
        
        # Step 3b: Get semantic embeddings for training features
        semantic_embeddings = None
        if os.getenv("USE_SAP_RPT_EMBEDDINGS", "false").lower() == "true":
            logger.info("Step 3b: Retrieving semantic embeddings...")
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
                graph_data, glean_data, learned_patterns, temporal_patterns, semantic_embeddings
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
        """Extract knowledge graph from source data via Extract service."""
        import httpx
        
        payload = {
            "json_tables": json_tables,
            "hive_ddls": hive_ddls,
            "sql_queries": [],
            "control_m_files": control_m_files,
            "signavio_files": signavio_files or [],
            "project_id": project_id,
            "system_id": system_id,
        }
        
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
        semantic_embeddings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate training features from graph data and Glean patterns."""
        features = []
        
        # Extract features from current knowledge graph
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
        
        return {
            "features": features,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "has_historical_data": glean_data is not None,
            "has_learned_patterns": learned_patterns is not None,
            "has_temporal_patterns": temporal_patterns is not None,
        }
    
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
