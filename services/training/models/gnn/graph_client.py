"""Graph service client for training pipeline.

This module provides a client to interact with the Graph service
for optimized Neo4j queries and batch operations.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Iterator
import httpx

from .api.graph_data_models import GraphData, Node, Edge, from_neo4j

logger = logging.getLogger(__name__)


class GraphServiceClient:
    """Client for Graph service Neo4j operations.
    
    Provides optimized access to Neo4j through the Graph service,
    leveraging batch operations and connection pooling.
    """
    
    def __init__(self, graph_service_url: Optional[str] = None, extract_service_url: Optional[str] = None):
        """Initialize Graph service client.
        
        Args:
            graph_service_url: Graph service URL (optional, for future direct Neo4j access)
            extract_service_url: Extract service URL (used for Neo4j queries via graph service workflows)
        """
        self.graph_service_url = graph_service_url or os.getenv(
            "GRAPH_SERVICE_URL", "http://graph-service:8081"
        )
        # Extract service is used by graph service for Neo4j queries
        self.extract_service_url = extract_service_url or os.getenv(
            "EXTRACT_SERVICE_URL", "http://extract-service:19080"
        )
        self.client = httpx.Client(timeout=300.0)  # Long timeout for graph processing
        
        # Feature flag for graph service integration
        self.use_graph_service = os.getenv("ENABLE_GRAPH_SERVICE_INTEGRATION", "true").lower() == "true"
    
    def query_neo4j(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a Cypher query against Neo4j.
        
        Phase 1: Uses graph service workflow for optimized Neo4j access when available,
        falls back to extract service for backward compatibility.
        This provides optimized batch operations through the graph service's connection pooling.
        
        Args:
            cypher: Cypher query string
            params: Optional query parameters
        
        Returns:
            Query results with columns and data in unified GraphData format
        """
        # Phase 1: Try graph service workflow endpoint first (if available)
        if self.use_graph_service and self.graph_service_url:
            try:
                # Use graph service's knowledge graph query workflow
                # This leverages graph service's optimized Neo4j client
                endpoint = f"{self.graph_service_url}/knowledge-graph/query"
                
                payload = {
                    "knowledge_graph_query": cypher,
                    "knowledge_graph_query_params": params or {}
                }
                
                logger.info(f"Executing Cypher query via graph service: {cypher[:100]}...")
                
                response = self.client.post(endpoint, json=payload, timeout=300.0)
                response.raise_for_status()
                
                result = response.json()
                
                # Extract query result from graph service response
                if "knowledge_graph_query_result" in result:
                    query_result = result["knowledge_graph_query_result"]
                elif "query_result" in result:
                    query_result = result["query_result"]
                else:
                    query_result = result
                
                logger.info(f"Query returned {len(query_result.get('data', []))} results via graph service")
                
                # Convert to unified GraphData format
                graph_data = from_neo4j(query_result)
                
                return {
                    "columns": query_result.get("columns", []),
                    "data": query_result.get("data", []),
                    "graph_data": graph_data.dict() if graph_data else None
                }
            except Exception as e:
                logger.warning(f"Graph service query failed, falling back to extract service: {e}")
                # Fall through to extract service
        
        # Fallback to extract service (which graph service uses internally)
        endpoint = f"{self.extract_service_url}/knowledge-graph/query"
        
        payload = {"query": cypher}
        if params:
            payload["params"] = params
        
        logger.info(f"Executing Cypher query via extract service: {cypher[:100]}...")
        
        try:
            response = self.client.post(endpoint, json=payload, timeout=300.0)
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(f"Query returned {len(result.get('data', []))} results")
            
            # Convert to unified GraphData format
            graph_data = from_neo4j(result)
            
            return {
                "columns": result.get("columns", []),
                "data": result.get("data", []),
                "graph_data": graph_data.dict() if graph_data else None
            }
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Neo4j query HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Neo4j query request error: {e}")
            raise
    
    def get_graph_for_training(
        self,
        project_id: str,
        system_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> GraphData:
        """Get graph data for training purposes.
        
        Uses optimized batch queries to fetch nodes and edges efficiently.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
            filters: Optional filters (node_types, edge_types, etc.)
            limit: Optional limit on number of nodes/edges
        
        Returns:
            GraphData instance with nodes and edges
        """
        # Build Cypher query with filters
        where_clauses = ["n.project_id = $project_id"]
        params = {"project_id": project_id}
        
        if system_id:
            where_clauses.append("n.system_id = $system_id")
            params["system_id"] = system_id
        
        if filters:
            if "node_types" in filters:
                node_types = filters["node_types"]
                if isinstance(node_types, list) and node_types:
                    where_clauses.append("n.type IN $node_types")
                    params["node_types"] = node_types
        
        where_clause = " AND ".join(where_clauses)
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        # Query nodes
        nodes_query = f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN n.id AS id, n.type AS type, n.label AS label, 
               n.properties_json AS properties
        {limit_clause}
        """
        
        nodes_result = self.query_neo4j(nodes_query, params)
        
        # Query edges
        edges_query = f"""
        MATCH (source)-[r]->(target)
        WHERE source.project_id = $project_id
          AND ($system_id IS NULL OR source.system_id = $system_id)
        RETURN source.id AS source_id, target.id AS target_id,
               type(r) AS label, r.properties_json AS properties
        {limit_clause}
        """
        
        edges_result = self.query_neo4j(edges_query, params)
        
        # Convert to GraphData
        nodes = []
        edges = []
        
        # Extract nodes from result
        for row in nodes_result.get("data", []):
            if isinstance(row, list) and len(row) >= 4:
                node = Node(
                    id=str(row[0]) if row[0] else "",
                    type=str(row[1]) if row[1] else "unknown",
                    label=str(row[2]) if row[2] else None,
                    properties=row[3] if isinstance(row[3], dict) else {}
                )
                if node.id:
                    nodes.append(node)
        
        # Extract edges from result
        for row in edges_result.get("data", []):
            if isinstance(row, list) and len(row) >= 4:
                edge = Edge(
                    source=str(row[0]) if row[0] else "",
                    target=str(row[1]) if row[1] else "",
                    label=str(row[2]) if row[2] else None,
                    properties=row[3] if isinstance(row[3], dict) else {}
                )
                if edge.source and edge.target:
                    edges.append(edge)
        
        return GraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                "project_id": project_id,
                "system_id": system_id
            } if system_id else {"project_id": project_id}
        )
    
    def stream_nodes(
        self,
        project_id: str,
        system_id: Optional[str] = None,
        batch_size: int = 1000
    ) -> Iterator[List[Node]]:
        """Stream nodes in batches for memory-efficient processing.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
            batch_size: Number of nodes per batch
        
        Yields:
            Lists of Node objects
        """
        offset = 0
        
        while True:
            query = """
            MATCH (n)
            WHERE n.project_id = $project_id
              AND ($system_id IS NULL OR n.system_id = $system_id)
            RETURN n.id AS id, n.type AS type, n.label AS label,
                   n.properties_json AS properties
            ORDER BY n.id
            SKIP $offset
            LIMIT $batch_size
            """
            
            params = {
                "project_id": project_id,
                "system_id": system_id,
                "offset": offset,
                "batch_size": batch_size
            }
            
            result = self.query_neo4j(query, params)
            
            batch_nodes = []
            for row in result.get("data", []):
                if isinstance(row, list) and len(row) >= 4:
                    node = Node(
                        id=str(row[0]) if row[0] else "",
                        type=str(row[1]) if row[1] else "unknown",
                        label=str(row[2]) if row[2] else None,
                        properties=row[3] if isinstance(row[3], dict) else {}
                    )
                    if node.id:
                        batch_nodes.append(node)
            
            if not batch_nodes:
                break
            
            yield batch_nodes
            offset += batch_size
            
            # If we got fewer nodes than batch_size, we're done
            if len(batch_nodes) < batch_size:
                break
    
    def stream_edges(
        self,
        project_id: str,
        system_id: Optional[str] = None,
        batch_size: int = 1000
    ) -> Iterator[List[Edge]]:
        """Stream edges in batches for memory-efficient processing.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
            batch_size: Number of edges per batch
        
        Yields:
            Lists of Edge objects
        """
        offset = 0
        
        while True:
            query = """
            MATCH (source)-[r]->(target)
            WHERE source.project_id = $project_id
              AND ($system_id IS NULL OR source.system_id = $system_id)
            RETURN source.id AS source_id, target.id AS target_id,
                   type(r) AS label, r.properties_json AS properties
            ORDER BY source.id, target.id
            SKIP $offset
            LIMIT $batch_size
            """
            
            params = {
                "project_id": project_id,
                "system_id": system_id,
                "offset": offset,
                "batch_size": batch_size
            }
            
            result = self.query_neo4j(query, params)
            
            batch_edges = []
            for row in result.get("data", []):
                if isinstance(row, list) and len(row) >= 4:
                    edge = Edge(
                        source=str(row[0]) if row[0] else "",
                        target=str(row[1]) if row[1] else "",
                        label=str(row[2]) if row[2] else None,
                        properties=row[3] if isinstance(row[3], dict) else {}
                    )
                    if edge.source and edge.target:
                        batch_edges.append(edge)
            
            if not batch_edges:
                break
            
            yield batch_edges
            offset += batch_size
            
            # If we got fewer edges than batch_size, we're done
            if len(batch_edges) < batch_size:
                break
    
    def health_check(self) -> bool:
        """Check if Graph service (via extract service) is available.
        
        Returns:
            True if service is healthy, False otherwise
        """
        endpoint = f"{self.extract_service_url}/healthz"
        
        try:
            response = self.client.get(endpoint, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

