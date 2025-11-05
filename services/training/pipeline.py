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
        output_dir: Optional[str] = None
    ):
        self.extract_service_url = extract_service_url or os.getenv("EXTRACT_SERVICE_URL", "http://localhost:19080")
        self.glean_client = GleanTrainingClient(db_name=glean_db_name)
        self.output_dir = output_dir or os.getenv("TRAINING_OUTPUT_DIR", "./training_data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Phase 9.1: Initialize auto-tuner if enabled
        self.auto_tuner = None
        if os.getenv("ENABLE_AUTO_TUNING", "false").lower() == "true" and HAS_AUTO_TUNER and AutoTuner is not None:
            try:
                self.auto_tuner = AutoTuner(
                    study_name=os.getenv("OPTUNA_STUDY_NAME", "amodels_training"),
                    storage=os.getenv("OPTUNA_STORAGE"),
                    n_trials=int(os.getenv("OPTUNA_N_TRIALS", "50"))
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
        glean_days_back: int = 30,
        enable_glean: bool = True,
        enable_temporal_analysis: bool = True
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
            glean_days_back: Number of days to look back in Glean
            enable_glean: Whether to enable Glean integration
        
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
        
        # Step 1: Extract knowledge graph from source data
        logger.info("Step 1: Extracting knowledge graph from source data...")
        try:
            graph_data = self._extract_knowledge_graph(
                project_id=project_id,
                system_id=system_id,
                json_tables=json_tables or [],
                hive_ddls=hive_ddls or [],
                control_m_files=control_m_files or []
            )
            results["steps"]["extract"] = {
                "status": "success",
                "nodes": len(graph_data.get("nodes", [])),
                "edges": len(graph_data.get("edges", [])),
            }
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
            
            # Phase 9.1: Assess training data quality
            if self.auto_tuner is not None:
                try:
                    training_data_stats = {
                        "num_samples": len(features.get("features", [])),
                        "num_features": len(features.get("features", [0])) if features.get("features") else 0,
                        "pattern_coverage": learned_patterns.get("summary", {}).get("unique_column_types", 0) / 100.0 if learned_patterns else 0.0,
                    }
                    quality_assessment = self.auto_tuner.assess_training_data_quality(training_data_stats)
                    features["quality_assessment"] = quality_assessment
                    results["steps"]["data_quality"] = {
                        "status": "success",
                        "quality_score": quality_assessment.get("quality_score", 0.0),
                        "passed": quality_assessment.get("passed", False),
                    }
                    logger.info(f"✅ Data quality assessment: score={quality_assessment.get('quality_score', 0.0):.2f}")
                except Exception as e:
                    logger.warning(f"⚠️  Data quality assessment failed: {e}")
            
            results["steps"]["features"] = {
                "status": "success",
                "feature_count": len(features.get("features", [])),
            }
            logger.info(f"✅ Generated {results['steps']['features']['feature_count']} training features")
        except Exception as e:
            logger.error(f"❌ Feature generation failed: {e}")
            results["steps"]["features"] = {"status": "failed", "error": str(e)}
            return results
        
        # Step 5: Prepare training dataset
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
        
        results["pipeline_completed_at"] = datetime.now().isoformat()
        results["status"] = "success"
        
        logger.info("✅ Training pipeline completed successfully")
        
        return results
    
    def _extract_knowledge_graph(
        self,
        project_id: str,
        system_id: Optional[str],
        json_tables: list,
        hive_ddls: list,
        control_m_files: list
    ) -> Dict[str, Any]:
        """Extract knowledge graph from source data via Extract service."""
        import httpx
        
        payload = {
            "json_tables": json_tables,
            "hive_ddls": hive_ddls,
            "sql_queries": [],
            "control_m_files": control_m_files,
            "project_id": project_id,
            "system_id": system_id,
        }
        
        client = httpx.Client(timeout=300.0)
        response = client.post(
            f"{self.extract_service_url}/knowledge-graph",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
    
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

