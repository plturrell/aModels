"""Semantic features for training pipeline.

This module provides utilities to extract semantic features from embeddings
and use them in the training process.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class SemanticFeatureExtractor:
    """Extracts semantic features from embeddings for training."""
    
    def __init__(self, extract_service_url: Optional[str] = None):
        """Initialize semantic feature extractor.
        
        Args:
            extract_service_url: Extract service URL for semantic search
        """
        self.extract_service_url = extract_service_url
        self.extract_client = None
        if extract_service_url:
            try:
                from .extract_client import ExtractServiceClient
                self.extract_client = ExtractServiceClient(extract_service_url=extract_service_url)
            except Exception as e:
                logger.warning(f"Failed to create Extract client: {e}")
    
    def extract_semantic_features(
        self,
        graph_data: Dict[str, Any],
        use_classifications: bool = True
    ) -> Dict[str, Any]:
        """Extract semantic features from knowledge graph.
        
        Args:
            graph_data: Knowledge graph data
            use_classifications: Whether to use table classifications
        
        Returns:
            Dictionary of semantic features
        """
        features = {
            "semantic_search_results": {},
            "classification_features": {},
            "embedding_metadata": {},
        }
        
        nodes = graph_data.get("nodes", [])
        
        # Extract classification features
        if use_classifications:
            classification_features = self._extract_classification_features(nodes)
            features["classification_features"] = classification_features
        
        # Get semantic embeddings for tables
        if self.extract_client:
            semantic_results = self._get_semantic_embeddings(nodes)
            features["semantic_search_results"] = semantic_results
        
        return features
    
    def _extract_classification_features(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features based on table classifications.
        
        Args:
            nodes: List of graph nodes
        
        Returns:
            Classification-based features
        """
        classification_counts = defaultdict(int)
        classification_confidence_sum = defaultdict(float)
        classification_confidence_count = defaultdict(int)
        
        for node in nodes:
            if node.get("type") == "table":
                props = node.get("props", {})
                classification = props.get("table_classification")
                confidence = props.get("classification_confidence", 0.0)
                
                if classification:
                    classification_counts[classification] += 1
                    if isinstance(confidence, (int, float)):
                        classification_confidence_sum[classification] += confidence
                        classification_confidence_count[classification] += 1
        
        # Calculate average confidence per classification
        avg_confidence = {}
        for classification in classification_counts:
            if classification_confidence_count[classification] > 0:
                avg_confidence[classification] = (
                    classification_confidence_sum[classification] / 
                    classification_confidence_count[classification]
                )
        
        return {
            "classification_counts": dict(classification_counts),
            "average_confidence": avg_confidence,
            "total_classified": sum(classification_counts.values()),
        }
    
    def _get_semantic_embeddings(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get semantic embeddings for tables.
        
        Args:
            nodes: List of graph nodes
        
        Returns:
            Dictionary of semantic embedding search results
        """
        semantic_results = {}
        
        for node in nodes:
            if node.get("type") == "table":
                table_name = node.get("label", "")
                artifact_id = node.get("id", "")
                
                if table_name and self.extract_client:
                    try:
                        # Search for semantic embedding
                        results = self.extract_client.search_semantic(
                            query=f"table {table_name}",
                            artifact_type="table",
                            limit=1,
                            use_semantic=True,
                            use_hybrid_search=True
                        )
                        
                        if results and len(results) > 0:
                            semantic_results[artifact_id] = {
                                "table_name": table_name,
                                "search_score": results[0].get("score", 0.0),
                                "metadata": results[0].get("metadata", {}),
                                "embedding_type": results[0].get("metadata", {}).get("embedding_type"),
                            }
                    except Exception as e:
                        logger.warning(f"Failed to get semantic embedding for {table_name}: {e}")
                        continue
        
        return semantic_results
    
    def get_table_classifications_for_routing(
        self,
        project_id: str,
        system_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Get table classifications for workflow routing.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
        
        Returns:
            Dictionary mapping table names to classifications
        """
        if not self.extract_client:
            return {}
        
        try:
            classifications_data = self.extract_client.get_table_classifications(
                project_id=project_id,
                system_id=system_id
            )
            
            # Convert to simple name -> classification mapping
            routing_map = {}
            for table_name, data in classifications_data.items():
                classification = data.get("classification")
                if classification:
                    routing_map[table_name] = classification
            
            return routing_map
        
        except Exception as e:
            logger.warning(f"Failed to get table classifications: {e}")
            return {}

