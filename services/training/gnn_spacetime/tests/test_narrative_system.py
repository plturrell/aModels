"""Integration tests for unified narrative system across all task modes."""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from gnn_spacetime.narrative import (
    NarrativeNode, NarrativeEdge, Storyline, NarrativeGraph,
    NarrativeType, MultiPurposeNarrativeGNN
)
from gnn_spacetime.data.temporal_node import TemporalNode
from gnn_spacetime.data.temporal_edge import TemporalEdge


class TestNarrativeSystem(unittest.TestCase):
    """Test unified narrative system across all modes."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample narrative graph
        self.graph = self._create_sample_narrative_graph()
        self.current_time = 10.0
    
    def _create_sample_narrative_graph(self) -> NarrativeGraph:
        """Create a sample narrative graph for testing."""
        # Create nodes
        node1 = NarrativeNode(
            node_id="company_a",
            node_type="company",
            static_embedding=torch.randn(128) if HAS_TORCH else None,
            lifespan=(0.0, None),
            narrative_roles={"merger_story": {"role": "protagonist", "arc_phase": "rising"}},
            causal_influence={"merger_story": 0.8},
            explanatory_power=0.9
        )
        
        node2 = NarrativeNode(
            node_id="company_b",
            node_type="company",
            static_embedding=torch.randn(128) if HAS_TORCH else None,
            lifespan=(0.0, None),
            narrative_roles={"merger_story": {"role": "target", "arc_phase": "rising"}},
            causal_influence={"merger_story": 0.7},
            explanatory_power=0.8
        )
        
        node3 = NarrativeNode(
            node_id="cultural_resistance",
            node_type="factor",
            static_embedding=torch.randn(128) if HAS_TORCH else None,
            lifespan=(5.0, None),
            narrative_roles={"merger_story": {"role": "antagonist", "arc_phase": "climax"}},
            causal_influence={"merger_story": 0.6},
            explanatory_power=0.7
        )
        
        # Create edges
        edge1 = NarrativeEdge(
            source_id="company_a",
            target_id="company_b",
            relation_type="acquires",
            temporal_scope=(0.0, None),
            narrative_significance={"merger_story": 0.9},
            causal_strength=0.8,
            counterfactual_importance=0.9
        )
        
        edge2 = NarrativeEdge(
            source_id="cultural_resistance",
            target_id="company_a",
            relation_type="impedes",
            temporal_scope=(5.0, None),
            narrative_significance={"merger_story": 0.7},
            causal_strength=0.6,
            counterfactual_importance=0.7
        )
        
        # Create storyline
        storyline = Storyline(
            storyline_id="merger_story",
            theme="Corporate merger between Company A and Company B",
            narrative_type=NarrativeType.EXPLANATION,
            coherence_metrics={
                "plot_consistency": 0.8,
                "character_development": 0.7,
                "causal_plausibility": 0.9
            },
            explanatory_quality=0.85,
            causal_links=[
                ("company_a", "company_b", "acquires"),
                ("cultural_resistance", "company_a", "impedes")
            ],
            key_events=[
                {"time": 0.0, "node_id": "company_a", "description": "Merger announcement"},
                {"time": 5.0, "node_id": "cultural_resistance", "description": "Cultural resistance emerges"},
                {"time": 10.0, "node_id": "company_a", "description": "Implementation delays"}
            ]
        )
        
        # Create graph
        graph = NarrativeGraph(
            nodes=[node1, node2, node3],
            edges=[edge1, edge2],
            storylines={"merger_story": storyline}
        )
        
        return graph
    
    def test_explanatory_mode(self):
        """Test explanatory AI mode."""
        gnn = MultiPurposeNarrativeGNN(narrative_graph=self.graph)
        
        result = gnn.forward(
            self.graph,
            current_time=self.current_time,
            task_mode="explain",
            storyline_id="merger_story"
        )
        
        # Assertions
        self.assertEqual(result["task"], "explain")
        self.assertIn("explanation", result)
        self.assertIsInstance(result["explanation"], str)
        self.assertGreater(len(result["explanation"]), 0)
        self.assertIn("storyline_id", result)
        self.assertEqual(result["storyline_id"], "merger_story")
        self.assertIn("explanatory_quality", result)
        self.assertGreaterEqual(result["explanatory_quality"], 0.0)
        self.assertLessEqual(result["explanatory_quality"], 1.0)
    
    def test_prediction_mode(self):
        """Test causal prediction mode."""
        gnn = MultiPurposeNarrativeGNN(narrative_graph=self.graph)
        
        result = gnn.forward(
            self.graph,
            current_time=self.current_time,
            task_mode="predict",
            storyline_id="merger_story"
        )
        
        # Assertions
        self.assertEqual(result["task"], "predict")
        self.assertIn("prediction", result)
        self.assertIn("confidence", result["prediction"])
        self.assertIn("predicted_events", result["prediction"])
        self.assertIsInstance(result["prediction"]["predicted_events"], list)
        self.assertIn("storyline_id", result)
        self.assertIn("predictive_confidence", result)
    
    def test_anomaly_detection_mode(self):
        """Test anomaly detection mode."""
        gnn = MultiPurposeNarrativeGNN(narrative_graph=self.graph)
        
        result = gnn.forward(
            self.graph,
            current_time=self.current_time,
            task_mode="detect_anomalies",
            storyline_id="merger_story"
        )
        
        # Assertions
        self.assertEqual(result["task"], "detect_anomalies")
        self.assertIn("anomalies", result)
        self.assertIsInstance(result["anomalies"], list)
        self.assertIn("storyline_id", result)
        self.assertIn("anomaly_score", result)
    
    def test_mode_switching(self):
        """Test switching between modes maintains consistency."""
        gnn = MultiPurposeNarrativeGNN(narrative_graph=self.graph)
        
        # Run all modes
        explain_result = gnn.forward(self.graph, self.current_time, "explain", "merger_story")
        predict_result = gnn.forward(self.graph, self.current_time, "predict", "merger_story")
        anomaly_result = gnn.forward(self.graph, self.current_time, "detect_anomalies", "merger_story")
        
        # All should reference same storyline
        self.assertEqual(explain_result["storyline_id"], "merger_story")
        self.assertEqual(predict_result["storyline_id"], "merger_story")
        self.assertEqual(anomaly_result["storyline_id"], "merger_story")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        gnn = MultiPurposeNarrativeGNN(narrative_graph=self.graph)
        
        # Test with non-existent storyline
        result = gnn.forward(
            self.graph,
            current_time=self.current_time,
            task_mode="explain",
            storyline_id="nonexistent_story"
        )
        
        self.assertIn("error", result)
    
    def test_narrative_aware_message_passing(self):
        """Test narrative-aware message passing."""
        gnn = MultiPurposeNarrativeGNN(narrative_graph=self.graph)
        
        node1 = self.graph.nodes["company_a"]
        node2 = self.graph.nodes["company_b"]
        edge = self.graph.edges[0]
        storyline = self.graph.get_storyline("merger_story")
        
        # Test different modes
        weight_explain = gnn.narrative_aware_message_passing(
            node1, node2, edge, storyline, "explain", self.current_time
        )
        weight_predict = gnn.narrative_aware_message_passing(
            node1, node2, edge, storyline, "predict", self.current_time
        )
        weight_anomaly = gnn.narrative_aware_message_passing(
            node1, node2, edge, storyline, "detect_anomalies", self.current_time
        )
        
        # Weights should be valid
        self.assertGreaterEqual(weight_explain, 0.0)
        self.assertLessEqual(weight_explain, 1.0)
        self.assertGreaterEqual(weight_predict, 0.0)
        self.assertLessEqual(weight_predict, 1.0)
        self.assertGreaterEqual(weight_anomaly, 0.0)
        self.assertLessEqual(weight_anomaly, 1.0)


if __name__ == "__main__":
    unittest.main()

