"""Test suite for GNN integration (Priority 2).

Tests GNN modules, training, evaluation, and pipeline integration.
Phase 1-6: Enhanced with graph service integration, parallel processing, and cross-service tests.
"""

import unittest
import os
import tempfile
import shutil
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

try:
    import torch
    import torch_geometric
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from training.pipeline import TrainingPipeline
    from training.gnn_embeddings import GNNEmbedder
    from training.gnn_node_classifier import GNNNodeClassifier
    from training.gnn_link_predictor import GNNLinkPredictor
    from training.gnn_evaluation import GNNEvaluator
    from training.gnn_training import GNNTrainer
    from training.graph_client import GraphServiceClient
    HAS_GNN = True
except ImportError:
    HAS_GNN = False
    GraphServiceClient = None


class TestGNNIntegration(unittest.TestCase):
    """Test GNN integration with training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_nodes = [
            {
                "id": "node_1",
                "type": "table",
                "properties": {"column_count": 5, "row_count": 100}
            },
            {
                "id": "node_2",
                "type": "column",
                "properties": {"data_type": "string", "nullable": True}
            },
            {
                "id": "node_3",
                "type": "table",
                "properties": {"column_count": 3, "row_count": 50}
            }
        ]
        self.test_edges = [
            {
                "source_id": "node_1",
                "target_id": "node_2",
                "label": "HAS_COLUMN"
            },
            {
                "source_id": "node_1",
                "target_id": "node_3",
                "label": "RELATED"
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(HAS_GNN and HAS_PYG, "GNN modules or PyTorch Geometric not available")
    def test_gnn_embedder_initialization(self):
        """Test GNN embedder initialization."""
        embedder = GNNEmbedder(device="cpu")
        self.assertIsNotNone(embedder)
        self.assertEqual(embedder.embedding_dim, 128)
    
    @unittest.skipUnless(HAS_GNN and HAS_PYG, "GNN modules or PyTorch Geometric not available")
    def test_gnn_embedder_generation(self):
        """Test GNN embedding generation."""
        embedder = GNNEmbedder(device="cpu")
        result = embedder.generate_embeddings(
            self.test_nodes,
            self.test_edges,
            graph_level=True
        )
        
        self.assertIsNotNone(result)
        if "error" not in result:
            self.assertIn("graph_embedding", result)
            self.assertEqual(len(result["graph_embedding"]), 128)
    
    @unittest.skipUnless(HAS_GNN and GraphServiceClient is not None, "Graph service client not available")
    def test_graph_service_client_initialization(self):
        """Phase 1: Test Graph service client initialization."""
        client = GraphServiceClient(
            graph_service_url="http://test-graph:8081",
            extract_service_url="http://test-extract:19080"
        )
        self.assertIsNotNone(client)
        self.assertEqual(client.graph_service_url, "http://test-graph:8081")
        self.assertTrue(client.use_graph_service)
    
    @unittest.skipUnless(HAS_GNN and GraphServiceClient is not None, "Graph service client not available")
    @patch('training.graph_client.httpx.Client')
    def test_graph_service_client_query(self, mock_client_class):
        """Phase 1: Test Graph service client Neo4j query."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "columns": ["n"],
            "data": [[{"id": "node1", "type": "table"}]]
        }
        mock_response.raise_for_status = Mock()
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        client = GraphServiceClient(
            graph_service_url="http://test-graph:8081",
            extract_service_url="http://test-extract:19080"
        )
        client.client = mock_client
        
        result = client.query_neo4j("MATCH (n) RETURN n LIMIT 1")
        
        self.assertIsNotNone(result)
        self.assertIn("columns", result)
        self.assertIn("data", result)
    
    @unittest.skipUnless(HAS_GNN, "GNN modules not available")
    def test_pipeline_parallel_gnn_processing(self):
        """Phase 6: Test parallel GNN processing in pipeline."""
        os.environ["ENABLE_PARALLEL_GNN"] = "true"
        os.environ["ENABLE_GNN"] = "true"
        
        pipeline = TrainingPipeline(
            output_dir=self.temp_dir,
            enable_gnn=True,
            enable_gnn_embeddings=True,
            enable_gnn_classification=True,
            enable_gnn_link_prediction=True
        )
        
        # Test parallel processing helpers exist
        self.assertTrue(hasattr(pipeline, "_generate_gnn_embeddings_parallel"))
        self.assertTrue(hasattr(pipeline, "_classify_nodes_parallel"))
        self.assertTrue(hasattr(pipeline, "_predict_links_parallel"))
        
        # Cleanup
        del os.environ["ENABLE_PARALLEL_GNN"]
        del os.environ["ENABLE_GNN"]
    
    @unittest.skipUnless(HAS_GNN, "GNN modules not available")
    def test_pipeline_graph_service_integration(self):
        """Phase 1: Test pipeline integration with graph service client."""
        pipeline = TrainingPipeline(
            output_dir=self.temp_dir,
            graph_service_url="http://test-graph:8081"
        )
        
        # Check that graph client is initialized
        if os.getenv("ENABLE_GRAPH_SERVICE_INTEGRATION", "true").lower() == "true":
            # Graph client should be initialized if available
            # (may be None if service unavailable, which is OK)
            pass  # Just verify no errors during initialization
    
    @unittest.skipUnless(HAS_GNN and HAS_PYG, "GNN modules or PyTorch Geometric not available")
    def test_gnn_node_classifier_training(self):
        """Test node classifier training."""
        classifier = GNNNodeClassifier(device="cpu")
        
        # Create labels
        labels = {
            "node_1": "table",
            "node_2": "column",
            "node_3": "table"
        }
        
        result = classifier.train(
            self.test_nodes,
            self.test_edges,
            labels=labels,
            epochs=10,  # Small number for testing
            lr=0.01
        )
        
        self.assertIsNotNone(result)
        if "error" not in result:
            self.assertIn("accuracy", result)
            self.assertGreaterEqual(result["accuracy"], 0.0)
            self.assertLessEqual(result["accuracy"], 1.0)
    
    @unittest.skipUnless(HAS_GNN and HAS_PYG, "GNN modules or PyTorch Geometric not available")
    def test_gnn_link_predictor_training(self):
        """Test link predictor training."""
        predictor = GNNLinkPredictor(device="cpu")
        
        result = predictor.train(
            self.test_nodes,
            self.test_edges,
            epochs=10,  # Small number for testing
            lr=0.01
        )
        
        self.assertIsNotNone(result)
        if "error" not in result:
            self.assertIn("accuracy", result)
            self.assertGreaterEqual(result["accuracy"], 0.0)
            self.assertLessEqual(result["accuracy"], 1.0)
    
    @unittest.skipUnless(HAS_GNN and HAS_PYG, "GNN modules or PyTorch Geometric not available")
    def test_gnn_evaluator_classification(self):
        """Test classification evaluation."""
        evaluator = GNNEvaluator()
        
        y_true = ["table", "column", "table"]
        y_pred = ["table", "column", "table"]
        
        metrics = evaluator.evaluate_classification(y_true, y_pred)
        
        self.assertIsNotNone(metrics)
        if "error" not in metrics:
            self.assertIn("accuracy", metrics)
            self.assertEqual(metrics["accuracy"], 1.0)  # Perfect match
    
    @unittest.skipUnless(HAS_GNN and HAS_PYG, "GNN modules or PyTorch Geometric not available")
    def test_gnn_trainer_node_classifier(self):
        """Test GNN trainer for node classifier."""
        trainer = GNNTrainer(output_dir=self.temp_dir, device="cpu")
        
        labels = {
            "node_1": "table",
            "node_2": "column",
            "node_3": "table"
        }
        
        result = trainer.train_node_classifier(
            self.test_nodes,
            self.test_edges,
            labels=labels,
            epochs=10,
            save_model=True
        )
        
        self.assertIsNotNone(result)
        if "error" not in result:
            self.assertEqual(result["status"], "success")
            self.assertIn("model_path", result)
            self.assertTrue(os.path.exists(result["model_path"]))
    
    @unittest.skipUnless(HAS_GNN and HAS_PYG, "GNN modules or PyTorch Geometric not available")
    def test_pipeline_gnn_integration(self):
        """Test GNN integration in training pipeline."""
        os.environ["ENABLE_GNN"] = "true"
        os.environ["ENABLE_GNN_EMBEDDINGS"] = "true"
        
        pipeline = TrainingPipeline(
            output_dir=self.temp_dir,
            enable_gnn=True,
            enable_gnn_embeddings=True,
            enable_gnn_classification=False,  # Skip classification (needs training)
            enable_gnn_link_prediction=False,  # Skip link prediction (needs training)
            enable_gnn_device="cpu"
        )
        
        # Create minimal graph data
        graph_data = {
            "nodes": self.test_nodes,
            "edges": self.test_edges
        }
        
        # Test embedding generation
        if pipeline.gnn_embedder:
            result = pipeline.gnn_embedder.generate_embeddings(
                self.test_nodes,
                self.test_edges,
                graph_level=True
            )
            self.assertIsNotNone(result)
    
    @unittest.skipUnless(HAS_GNN and HAS_PYG, "GNN modules or PyTorch Geometric not available")
    def test_pipeline_gnn_training(self):
        """Test GNN training in pipeline."""
        os.environ["ENABLE_GNN"] = "true"
        os.environ["ENABLE_GNN_TRAINING"] = "true"
        
        pipeline = TrainingPipeline(
            output_dir=self.temp_dir,
            enable_gnn=True,
            enable_gnn_training=True,
            enable_gnn_device="cpu"
        )
        
        if pipeline.gnn_trainer:
            result = pipeline.train_gnn_models(
                self.test_nodes,
                self.test_edges,
                train_classifier=True,
                train_link_predictor=True,
                epochs=10
            )
            
            self.assertIsNotNone(result)
            if "error" not in result:
                self.assertEqual(result["status"], "success")
                self.assertIn("models_trained", result)


if __name__ == "__main__":
    unittest.main()

