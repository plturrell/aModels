"""Unified data access layer for training service.

Improvement 4: Provides a unified interface for accessing data from all storage systems
(Postgres, Redis, Neo4j) with consistent query patterns and caching.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Iterator
from abc import ABC, abstractmethod
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from neo4j import GraphDatabase

from .api.graph_data_models import GraphData, Node, Edge

logger = logging.getLogger(__name__)


class StorageAdapter(ABC):
    """Abstract base class for storage system adapters."""
    
    @abstractmethod
    def get_nodes(self, project_id: str, system_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Node]:
        """Get nodes from storage."""
        pass
    
    @abstractmethod
    def get_edges(self, project_id: str, system_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Edge]:
        """Get edges from storage."""
        pass
    
    @abstractmethod
    def count_nodes(self, project_id: str, system_id: Optional[str] = None) -> int:
        """Count nodes in storage."""
        pass
    
    @abstractmethod
    def count_edges(self, project_id: str, system_id: Optional[str] = None) -> int:
        """Count edges in storage."""
        pass


class PostgresAdapter(StorageAdapter):
    """Adapter for Postgres storage."""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn = None
    
    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn, cursor_factory=RealDictCursor)
        return self._conn
    
    def get_nodes(self, project_id: str, system_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Node]:
        """Get nodes from Postgres."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        query = """
            SELECT id, kind as type, label, properties_json as properties
            FROM glean_nodes
            WHERE properties_json->>'project_id' = %s
        """
        params = [project_id]
        
        if system_id:
            query += " AND properties_json->>'system_id' = %s"
            params.append(system_id)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        nodes = []
        for row in rows:
            nodes.append(Node(
                id=str(row['id']),
                type=str(row['type']) if row['type'] else 'unknown',
                label=row['label'],
                properties=row['properties'] if isinstance(row['properties'], dict) else {}
            ))
        
        cur.close()
        return nodes
    
    def get_edges(self, project_id: str, system_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Edge]:
        """Get edges from Postgres."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        query = """
            SELECT e.source_id, e.target_id, e.label, e.properties_json as properties
            FROM glean_edges e
            JOIN glean_nodes n1 ON e.source_id = n1.id
            WHERE n1.properties_json->>'project_id' = %s
        """
        params = [project_id]
        
        if system_id:
            query += " AND n1.properties_json->>'system_id' = %s"
            params.append(system_id)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        edges = []
        for row in rows:
            edges.append(Edge(
                source_id=str(row['source_id']),
                target_id=str(row['target_id']),
                label=row['label'],
                properties=row['properties'] if isinstance(row['properties'], dict) else {}
            ))
        
        cur.close()
        return edges
    
    def count_nodes(self, project_id: str, system_id: Optional[str] = None) -> int:
        """Count nodes in Postgres."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        query = "SELECT COUNT(*) FROM glean_nodes WHERE properties_json->>'project_id' = %s"
        params = [project_id]
        
        if system_id:
            query += " AND properties_json->>'system_id' = %s"
            params.append(system_id)
        
        cur.execute(query, params)
        count = cur.fetchone()[0]
        cur.close()
        
        return count
    
    def count_edges(self, project_id: str, system_id: Optional[str] = None) -> int:
        """Count edges in Postgres."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        query = """
            SELECT COUNT(*) FROM glean_edges e
            JOIN glean_nodes n1 ON e.source_id = n1.id
            WHERE n1.properties_json->>'project_id' = %s
        """
        params = [project_id]
        
        if system_id:
            query += " AND n1.properties_json->>'system_id' = %s"
            params.append(system_id)
        
        cur.execute(query, params)
        count = cur.fetchone()[0]
        cur.close()
        
        return count


class RedisAdapter(StorageAdapter):
    """Adapter for Redis storage."""
    
    def __init__(self, redis_url: str):
        # Parse Redis URL
        if redis_url.startswith("redis://"):
            redis_url = redis_url.replace("redis://", "")
        
        parts = redis_url.split("/")
        host_port = parts[0].split(":")
        host = host_port[0] if len(host_port) > 0 else "localhost"
        port = int(host_port[1]) if len(host_port) > 1 else 6379
        db = int(parts[1]) if len(parts) > 1 else 0
        
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    def get_nodes(self, project_id: str, system_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Node]:
        """Get nodes from Redis."""
        # Redis stores schema data, but full node retrieval would need to reconstruct from keys
        # For now, return empty list as Redis is primarily used for caching
        return []
    
    def get_edges(self, project_id: str, system_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Edge]:
        """Get edges from Redis."""
        # Redis stores schema data, but full edge retrieval would need to reconstruct from keys
        # For now, return empty list as Redis is primarily used for caching
        return []
    
    def count_nodes(self, project_id: str, system_id: Optional[str] = None) -> int:
        """Count nodes in Redis."""
        keys = list(self.client.scan_iter(match="schema:node:*", count=1000))
        return len(keys)
    
    def count_edges(self, project_id: str, system_id: Optional[str] = None) -> int:
        """Count edges in Redis."""
        keys = list(self.client.scan_iter(match="schema:edge:*", count=1000))
        return len(keys)


class Neo4jAdapter(StorageAdapter):
    """Adapter for Neo4j storage."""
    
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def get_nodes(self, project_id: str, system_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Node]:
        """Get nodes from Neo4j."""
        with self.driver.session() as session:
            query = "MATCH (n:Node) WHERE n.properties_json CONTAINS $project_id RETURN n"
            params = {"project_id": project_id}
            
            if system_id:
                query += " AND n.properties_json CONTAINS $system_id"
                params["system_id"] = system_id
            
            result = session.run(query, params)
            
            nodes = []
            for record in result:
                node_data = dict(record["n"])
                nodes.append(Node(
                    id=str(node_data.get("id", "")),
                    type=str(node_data.get("type", "unknown")),
                    label=node_data.get("label"),
                    properties=node_data.get("properties_json", {}) if isinstance(node_data.get("properties_json"), dict) else {}
                ))
            
            return nodes
    
    def get_edges(self, project_id: str, system_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Edge]:
        """Get edges from Neo4j."""
        with self.driver.session() as session:
            query = """
                MATCH (source:Node)-[r:RELATIONSHIP]->(target:Node)
                WHERE source.properties_json CONTAINS $project_id
                RETURN source.id as source_id, target.id as target_id, r.label as label, r.properties_json as properties
            """
            params = {"project_id": project_id}
            
            result = session.run(query, params)
            
            edges = []
            for record in result:
                edges.append(Edge(
                    source_id=str(record["source_id"]),
                    target_id=str(record["target_id"]),
                    label=record.get("label"),
                    properties=record.get("properties", {}) if isinstance(record.get("properties"), dict) else {}
                ))
            
            return edges
    
    def count_nodes(self, project_id: str, system_id: Optional[str] = None) -> int:
        """Count nodes in Neo4j."""
        with self.driver.session() as session:
            query = "MATCH (n:Node) WHERE n.properties_json CONTAINS $project_id RETURN COUNT(n) as count"
            params = {"project_id": project_id}
            
            result = session.run(query, params)
            record = result.single()
            return record["count"] if record else 0
    
    def count_edges(self, project_id: str, system_id: Optional[str] = None) -> int:
        """Count edges in Neo4j."""
        with self.driver.session() as session:
            query = """
                MATCH (source:Node)-[r:RELATIONSHIP]->(target:Node)
                WHERE source.properties_json CONTAINS $project_id
                RETURN COUNT(r) as count
            """
            params = {"project_id": project_id}
            
            result = session.run(query, params)
            record = result.single()
            return record["count"] if record else 0


class UnifiedDataAccess:
    """Unified data access layer for all storage systems.
    
    Improvement 4: Provides a single interface for accessing data from Postgres, Redis, and Neo4j
    with automatic caching and query optimization.
    """
    
    def __init__(self, 
                 postgres_dsn: Optional[str] = None,
                 redis_url: Optional[str] = None,
                 neo4j_uri: Optional[str] = None,
                 neo4j_username: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 cache_manager=None):
        """Initialize unified data access layer.
        
        Args:
            postgres_dsn: Postgres connection string
            redis_url: Redis connection URL
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            cache_manager: Optional cache manager for query caching
        """
        self.adapters = {}
        self.cache_manager = cache_manager
        
        # Initialize adapters based on available connections
        if postgres_dsn:
            try:
                self.adapters['postgres'] = PostgresAdapter(postgres_dsn)
                logger.info("Postgres adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Postgres adapter: {e}")
        
        if redis_url:
            try:
                self.adapters['redis'] = RedisAdapter(redis_url)
                logger.info("Redis adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis adapter: {e}")
        
        if neo4j_uri:
            try:
                self.adapters['neo4j'] = Neo4jAdapter(
                    neo4j_uri,
                    neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j"),
                    neo4j_password or os.getenv("NEO4J_PASSWORD", "amodels123")
                )
                logger.info("Neo4j adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j adapter: {e}")
        
        # Primary storage system (prefer Neo4j, fallback to Postgres)
        self.primary_storage = 'neo4j' if 'neo4j' in self.adapters else ('postgres' if 'postgres' in self.adapters else None)
    
    def get_graph_data(self, 
                      project_id: str, 
                      system_id: Optional[str] = None,
                      filters: Optional[Dict[str, Any]] = None,
                      use_cache: bool = True) -> GraphData:
        """Get graph data from primary storage system with caching.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
            filters: Optional filters
            use_cache: Whether to use cache
        
        Returns:
            GraphData object with nodes and edges
        """
        if not self.primary_storage:
            raise ValueError("No storage adapters available")
        
        # Check cache first
        cache_key = f"graph:{project_id}:{system_id or 'all'}"
        if use_cache and self.cache_manager:
            cached = self.cache_manager.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return cached
        
        # Get data from primary storage
        adapter = self.adapters[self.primary_storage]
        nodes = adapter.get_nodes(project_id, system_id, filters)
        edges = adapter.get_edges(project_id, system_id, filters)
        
        graph_data = GraphData(nodes=nodes, edges=edges)
        
        # Cache the result
        if use_cache and self.cache_manager:
            self.cache_manager.set(cache_key, graph_data)
        
        return graph_data
    
    def get_nodes(self, 
                  project_id: str, 
                  system_id: Optional[str] = None,
                  filters: Optional[Dict[str, Any]] = None,
                  storage: Optional[str] = None) -> List[Node]:
        """Get nodes from specified or primary storage.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
            filters: Optional filters
            storage: Storage system to use ('postgres', 'redis', 'neo4j') or None for primary
        
        Returns:
            List of Node objects
        """
        storage = storage or self.primary_storage
        if storage not in self.adapters:
            raise ValueError(f"Storage adapter '{storage}' not available")
        
        return self.adapters[storage].get_nodes(project_id, system_id, filters)
    
    def get_edges(self, 
                  project_id: str, 
                  system_id: Optional[str] = None,
                  filters: Optional[Dict[str, Any]] = None,
                  storage: Optional[str] = None) -> List[Edge]:
        """Get edges from specified or primary storage.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
            filters: Optional filters
            storage: Storage system to use ('postgres', 'redis', 'neo4j') or None for primary
        
        Returns:
            List of Edge objects
        """
        storage = storage or self.primary_storage
        if storage not in self.adapters:
            raise ValueError(f"Storage adapter '{storage}' not available")
        
        return self.adapters[storage].get_edges(project_id, system_id, filters)
    
    def validate_consistency(self, project_id: str, system_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate data consistency across all storage systems.
        
        Args:
            project_id: Project ID
            system_id: Optional system ID
        
        Returns:
            Dictionary with consistency metrics and issues
        """
        counts = {}
        
        for name, adapter in self.adapters.items():
            try:
                node_count = adapter.count_nodes(project_id, system_id)
                edge_count = adapter.count_edges(project_id, system_id)
                counts[name] = {
                    'nodes': node_count,
                    'edges': edge_count
                }
            except Exception as e:
                logger.warning(f"Failed to get counts from {name}: {e}")
                counts[name] = {'nodes': 0, 'edges': 0}
        
        # Calculate variance
        node_counts = [c['nodes'] for c in counts.values()]
        edge_counts = [c['edges'] for c in counts.values()]
        
        node_variance = max(node_counts) - min(node_counts) if node_counts else 0
        edge_variance = max(edge_counts) - min(edge_counts) if edge_counts else 0
        
        issues = []
        if node_variance > 0:
            max_nodes = max(node_counts) if node_counts else 0
            variance_pct = (node_variance / max_nodes * 100) if max_nodes > 0 else 0
            if variance_pct > 5:
                issues.append({
                    'type': 'node_count_mismatch',
                    'severity': 'high' if variance_pct > 20 else 'medium',
                    'variance': node_variance,
                    'variance_pct': variance_pct
                })
        
        if edge_variance > 0:
            max_edges = max(edge_counts) if edge_counts else 0
            variance_pct = (edge_variance / max_edges * 100) if max_edges > 0 else 0
            if variance_pct > 5:
                issues.append({
                    'type': 'edge_count_mismatch',
                    'severity': 'high' if variance_pct > 20 else 'medium',
                    'variance': edge_variance,
                    'variance_pct': variance_pct
                })
        
        return {
            'consistent': len(issues) == 0,
            'counts': counts,
            'node_variance': node_variance,
            'edge_variance': edge_variance,
            'issues': issues
        }

