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
    
    def search_semantic(
        self,
        query: str,
        artifact_type: Optional[str] = None,
        limit: int = 10,
        use_semantic: bool = True,
        use_hybrid_search: bool = False
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using sap-rpt-1-oss embeddings.
        
        Args:
            query: Search query (natural language)
            artifact_type: Optional artifact type filter (table, column, etc.)
            limit: Maximum number of results
            use_semantic: Use semantic embeddings (sap-rpt-1-oss)
            use_hybrid_search: Search both relational and semantic embeddings
        
        Returns:
            List of search results with metadata
        """
        endpoint = f"{self.extract_service_url}/knowledge-graph/search"
        
        payload = {
            "query": query,
            "limit": limit,
            "use_semantic": use_semantic,
            "use_hybrid_search": use_hybrid_search,
        }
        
        if artifact_type:
            payload["artifact_type"] = artifact_type
        
        logger.info(f"Performing semantic search: {query[:100]}...")
        
        try:
            response = self.client.post(endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            results = result.get("results", [])
            logger.info(f"Semantic search returned {len(results)} results")
            
            return results
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Semantic search HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Semantic search request error: {e}")
            raise
    
    def get_table_classifications(
        self,
        project_id: str,
        system_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get table classifications for a project/system.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
        
        Returns:
            Dictionary mapping table names to classification data
        """
        query = """
        MATCH (n)
        WHERE n.type = 'table'
          AND n.props.project_id = $project_id
          AND ($system_id IS NULL OR n.props.system_id = $system_id)
          AND n.props.table_classification IS NOT NULL
        RETURN n.label AS table_name, 
               n.props.table_classification AS classification,
               n.props.classification_confidence AS confidence,
               n.props.classification_evidence AS evidence
        """
        
        params = {"project_id": project_id}
        if system_id:
            params["system_id"] = system_id
        
        result = self.query_knowledge_graph(query, params)
        
        classifications = {}
        for row in result.get("data", []):
            table_name = row.get("table_name")
            if table_name:
                classifications[table_name] = {
                    "classification": row.get("classification"),
                    "confidence": row.get("confidence"),
                    "evidence": row.get("evidence"),
                }
        
        return classifications
    
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

