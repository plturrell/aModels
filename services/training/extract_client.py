"""Extract service client for training pipeline.

This module provides a client to interact with the Extract service
to get knowledge graphs and integrate them into training.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import httpx

logger = logging.getLogger(__name__)


class ExtractServiceClient:
    """Client for interacting with the Extract service."""
    
    def __init__(self, extract_service_url: Optional[str] = None):
        self.extract_service_url = extract_service_url or os.getenv(
            "EXTRACT_SERVICE_URL", "http://localhost:19080"
        )
        self.client = httpx.Client(timeout=300.0)  # Long timeout for graph processing
    
    def get_knowledge_graph(
        self,
        project_id: str,
        system_id: Optional[str] = None,
        json_tables: Optional[List[str]] = None,
        hive_ddls: Optional[List[str]] = None,
        sql_queries: Optional[List[str]] = None,
        control_m_files: Optional[List[str]] = None,
        ideal_distribution: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Get knowledge graph from Extract service.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
            json_tables: List of JSON table file paths
            hive_ddls: List of Hive DDL file paths
            sql_queries: List of SQL query strings
            control_m_files: List of Control-M XML file paths
            ideal_distribution: Optional ideal data type distribution
        
        Returns:
            Dictionary with knowledge graph:
            - nodes: List of graph nodes
            - edges: List of graph edges
            - metrics: Information theory metrics
            - quality: Quality assessment
        """
        endpoint = f"{self.extract_service_url}/knowledge-graph"
        
        payload = {
            "project_id": project_id,
        }
        
        if system_id:
            payload["system_id"] = system_id
        if json_tables:
            payload["json_tables"] = json_tables
        if hive_ddls:
            payload["hive_ddls"] = hive_ddls
        if sql_queries:
            payload["sql_queries"] = sql_queries
        if control_m_files:
            payload["control_m_files"] = control_m_files
        if ideal_distribution:
            payload["ideal_distribution"] = ideal_distribution
        
        logger.info(f"Requesting knowledge graph from Extract service: {endpoint}")
        
        try:
            response = self.client.post(endpoint, json=payload)
            response.raise_for_status()
            
            graph_data = response.json()
            
            logger.info(
                f"Received knowledge graph: {len(graph_data.get('nodes', []))} nodes, "
                f"{len(graph_data.get('edges', []))} edges"
            )
            
            return graph_data
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Extract service HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Extract service request error: {e}")
            raise
    
    def query_knowledge_graph(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a Cypher query against the Neo4j knowledge graph.
        
        Args:
            query: Cypher query string
            params: Optional query parameters
        
        Returns:
            Query results with columns and data
        """
        endpoint = f"{self.extract_service_url}/knowledge-graph/query"
        
        payload = {"query": query}
        if params:
            payload["params"] = params
        
        logger.info(f"Executing Cypher query: {query[:100]}...")
        
        try:
            response = self.client.post(endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(f"Query returned {len(result.get('data', []))} results")
            
            return result
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Neo4j query HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Neo4j query request error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Extract service is available.
        
        Returns:
            True if service is healthy, False otherwise
        """
        endpoint = f"{self.extract_service_url}/healthz"
        
        try:
            response = self.client.get(endpoint, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

