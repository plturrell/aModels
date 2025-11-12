"""Glean Catalog integration for training pipeline.

This module provides functions to query Glean Catalog for historical patterns
and integrate them into the training pipeline.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Glean configuration
GLEAN_DB_NAME = os.getenv("GLEAN_DB_NAME", "amodels")
GLEAN_SCHEMA_PATH = os.getenv("GLEAN_SCHEMA_PATH", "")
GLEAN_QUERY_API_URL = os.getenv("GLEAN_QUERY_API_URL", "")
GLEAN_USE_CLI = os.getenv("GLEAN_USE_CLI", "true").lower() == "true"


class GleanTrainingClient:
    """Client for querying Glean Catalog for training data."""
    
    def __init__(self, db_name: Optional[str] = None, schema_path: Optional[str] = None):
        self.db_name = db_name or GLEAN_DB_NAME
        self.schema_path = schema_path or GLEAN_SCHEMA_PATH
        self.use_cli = GLEAN_USE_CLI
        self.api_url = GLEAN_QUERY_API_URL
        
    def query_historical_nodes(
        self,
        project_id: Optional[str] = None,
        system_id: Optional[str] = None,
        days_back: int = 30,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query historical node data from Glean Catalog.
        
        Args:
            project_id: Optional project ID filter
            system_id: Optional system ID filter
            days_back: Number of days to look back (default: 30)
            limit: Maximum number of results (default: 1000)
        
        Returns:
            List of node facts from Glean
        """
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        query_parts = [f'agenticAiETH.ETL.Node.1']
        where_clauses = []
        
        if project_id:
            where_clauses.append(f'project_id = "{project_id}"')
        if system_id:
            where_clauses.append(f'system_id = "{system_id}"')
        where_clauses.append(f'exported_at >= "{cutoff_date}"')
        
        if where_clauses:
            query_parts.append('where ' + ' and '.join(where_clauses))
        
        query = ' '.join(query_parts)
        
        return self._execute_query(query, limit)
    
    def query_historical_edges(
        self,
        project_id: Optional[str] = None,
        system_id: Optional[str] = None,
        days_back: int = 30,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query historical edge data from Glean Catalog.
        
        Args:
            project_id: Optional project ID filter
            system_id: Optional system ID filter
            days_back: Number of days to look back (default: 30)
            limit: Maximum number of results (default: 1000)
        
        Returns:
            List of edge facts from Glean
        """
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        query_parts = [f'agenticAiETH.ETL.Edge.1']
        where_clauses = []
        
        if project_id:
            where_clauses.append(f'project_id = "{project_id}"')
        if system_id:
            where_clauses.append(f'system_id = "{system_id}"')
        where_clauses.append(f'exported_at >= "{cutoff_date}"')
        
        if where_clauses:
            query_parts.append('where ' + ' and '.join(where_clauses))
        
        query = ' '.join(query_parts)
        
        return self._execute_query(query, limit)
    
    def query_export_manifests(
        self,
        project_id: Optional[str] = None,
        system_id: Optional[str] = None,
        days_back: int = 30,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query historical export manifests for training metadata.
        
        Args:
            project_id: Optional project ID filter
            system_id: Optional system ID filter
            days_back: Number of days to look back (default: 30)
            limit: Maximum number of results (default: 100)
        
        Returns:
            List of export manifest facts with metrics
        """
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        query_parts = [f'agenticAiETH.ETL.ExportManifest.1']
        where_clauses = []
        
        if project_id:
            where_clauses.append(f'project_id = "{project_id}"')
        if system_id:
            where_clauses.append(f'system_id = "{system_id}"')
        where_clauses.append(f'exported_at >= "{cutoff_date}"')
        
        if where_clauses:
            query_parts.append('where ' + ' and '.join(where_clauses))
        
        query = ' '.join(query_parts)
        
        return self._execute_query(query, limit)
    
    def query_information_theory_metrics(
        self,
        project_id: Optional[str] = None,
        system_id: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Query information theory metrics over time for pattern analysis.
        
        Args:
            project_id: Optional project ID filter
            system_id: Optional system ID filter
            days_back: Number of days to look back (default: 30)
        
        Returns:
            Dictionary with temporal metrics:
            - metadata_entropy_trend: List of (date, entropy) tuples
            - kl_divergence_trend: List of (date, kl_div) tuples
            - column_count_trend: List of (date, count) tuples
            - averages: Average values over time period
        """
        manifests = self.query_export_manifests(
            project_id=project_id,
            system_id=system_id,
            days_back=days_back,
            limit=1000
        )
        
        metadata_entropy_trend = []
        kl_divergence_trend = []
        column_count_trend = []
        
        for manifest in manifests:
            key = manifest.get("key", {})
            exported_at = key.get("exported_at", "")
            metadata_entropy = key.get("metadata_entropy")
            kl_divergence = key.get("kl_divergence")
            column_count = key.get("column_count")
            
            if exported_at:
                if metadata_entropy is not None:
                    metadata_entropy_trend.append((exported_at, metadata_entropy))
                if kl_divergence is not None:
                    kl_divergence_trend.append((exported_at, kl_divergence))
                if column_count is not None:
                    column_count_trend.append((exported_at, column_count))
        
        # Calculate averages
        avg_entropy = sum(v for _, v in metadata_entropy_trend) / len(metadata_entropy_trend) if metadata_entropy_trend else None
        avg_kl_div = sum(v for _, v in kl_divergence_trend) / len(kl_divergence_trend) if kl_divergence_trend else None
        avg_col_count = sum(v for _, v in column_count_trend) / len(column_count_trend) if column_count_trend else None
        
        return {
            "metadata_entropy_trend": metadata_entropy_trend,
            "kl_divergence_trend": kl_divergence_trend,
            "column_count_trend": column_count_trend,
            "averages": {
                "metadata_entropy": avg_entropy,
                "kl_divergence": avg_kl_div,
                "column_count": avg_col_count,
            },
            "sample_count": len(manifests),
        }
    
    def query_column_type_patterns(
        self,
        project_id: Optional[str] = None,
        system_id: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Query column type distribution patterns over time.
        
        Args:
            project_id: Optional project ID filter
            system_id: Optional system ID filter
            days_back: Number of days to look back (default: 30)
        
        Returns:
            Dictionary with column type patterns:
            - type_distributions: List of (date, distribution) tuples
            - common_types: Most common column types
            - type_transitions: Patterns of type changes
        """
        nodes = self.query_historical_nodes(
            project_id=project_id,
            system_id=system_id,
            days_back=days_back,
            limit=5000
        )
        
        # Extract column type information from nodes
        type_distributions = {}
        type_counts = {}
        
        for node in nodes:
            key = node.get("key", {})
            node_type = key.get("kind", "")
            props = key.get("properties_json", {})
            
            if isinstance(props, str):
                try:
                    props = json.loads(props)
                except json.JSONDecodeError:
                    props = {}
            
            if node_type == "column" and isinstance(props, dict):
                column_type = props.get("type", props.get("data_type", "unknown"))
                exported_at = key.get("exported_at", "")
                
                if exported_at and column_type:
                    date_key = exported_at[:10]  # YYYY-MM-DD
                    if date_key not in type_distributions:
                        type_distributions[date_key] = {}
                    type_distributions[date_key][column_type] = type_distributions[date_key].get(column_type, 0) + 1
                    type_counts[column_type] = type_counts.get(column_type, 0) + 1
        
        # Convert to list of tuples
        type_distributions_list = [
            (date, dist) for date, dist in sorted(type_distributions.items())
        ]
        
        # Get most common types
        common_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "type_distributions": type_distributions_list,
            "common_types": dict(common_types),
            "total_columns": sum(type_counts.values()),
        }
    
    def _execute_query(self, query: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Execute a Glean query and return results.
        
        Args:
            query: Glean query string
            limit: Maximum number of results
        
        Returns:
            List of query results
        """
        if self.use_cli:
            return self._query_via_cli(query, limit)
        else:
            return self._query_via_api(query, limit)
    
    def _query_via_cli(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Query Glean using CLI command."""
        if not self.schema_path:
            schema_root = os.getenv("GLEAN_SCHEMA_ROOT", "")
            if schema_root:
                schema_path = os.path.join(schema_root, "source", "etl.angle")
            else:
                schema_path = "glean/schema/source/etl.angle"
        else:
            schema_path = self.schema_path
        
        query_str = query
        if limit > 0:
            query_str = f"{query_str} | limit {limit}"
        
        cmd = [
            "glean",
            "query",
            "--schema", schema_path,
            self.db_name,
            query_str,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60.0,
                check=False,
            )
            
            if result.returncode != 0:
                if "command not found" in result.stderr.lower():
                    logger.warning(f"Glean CLI not available. Install Glean CLI or set GLEAN_QUERY_API_URL for REST API.")
                    return []
                logger.warning(f"Glean query failed: {result.stderr}")
                return []
            
            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                return []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Glean query output as JSON")
                return []
        
        except FileNotFoundError:
            logger.warning("Glean CLI not found. Set GLEAN_QUERY_API_URL for REST API access.")
            return []
        except subprocess.TimeoutExpired:
            logger.warning(f"Glean query timed out after 60 seconds")
            return []
        except Exception as e:
            logger.warning(f"Error executing Glean CLI: {e}")
            return []
    
    def _query_via_api(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Query Glean using REST API."""
        import httpx
        
        if not self.api_url:
            logger.warning("Glean API URL not configured (GLEAN_QUERY_API_URL)")
            return []
        
        client = httpx.Client(timeout=60.0)
        
        try:
            payload = {
                "predicate": query.split()[0],  # Extract predicate name
                "query": query,
                "limit": limit,
            }
            
            response = client.post(
                f"{self.api_url}/query",
                json=payload,
            )
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
            return []
        
        except httpx.HTTPStatusError as e:
            logger.warning(f"Glean API error: HTTP {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.warning(f"Glean API request failed: {e}")
            return []


def ingest_glean_data_for_training(
    project_id: Optional[str] = None,
    system_id: Optional[str] = None,
    days_back: int = 30,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Ingest Glean data into training pipeline format.
    
    This function queries Glean Catalog for historical data and transforms it
    into a format suitable for training.
    
    Args:
        project_id: Optional project ID filter
        system_id: Optional system ID filter
        days_back: Number of days to look back (default: 30)
        output_dir: Optional directory to save training data files
    
    Returns:
        Dictionary with:
        - nodes: List of historical nodes
        - edges: List of historical edges
        - metrics: Information theory metrics over time
        - column_patterns: Column type distribution patterns
        - output_files: List of generated output files (if output_dir provided)
    """
    client = GleanTrainingClient()
    
    logger.info(f"Ingesting Glean data for training (project={project_id}, system={system_id}, days_back={days_back})")
    
    # Query historical data
    nodes = client.query_historical_nodes(
        project_id=project_id,
        system_id=system_id,
        days_back=days_back,
        limit=10000
    )
    
    edges = client.query_historical_edges(
        project_id=project_id,
        system_id=system_id,
        days_back=days_back,
        limit=10000
    )
    
    # Query metrics and patterns
    metrics = client.query_information_theory_metrics(
        project_id=project_id,
        system_id=system_id,
        days_back=days_back
    )
    
    column_patterns = client.query_column_type_patterns(
        project_id=project_id,
        system_id=system_id,
        days_back=days_back
    )
    
    result = {
        "nodes": nodes,
        "edges": edges,
        "metrics": metrics,
        "column_patterns": column_patterns,
        "metadata": {
            "project_id": project_id,
            "system_id": system_id,
            "days_back": days_back,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "ingested_at": datetime.now().isoformat(),
        }
    }
    
    # Save to files if output_dir provided
    output_files = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save nodes
        nodes_file = os.path.join(output_dir, "glean_nodes.json")
        with open(nodes_file, 'w') as f:
            json.dump(nodes, f, indent=2)
        output_files.append(nodes_file)
        
        # Save edges
        edges_file = os.path.join(output_dir, "glean_edges.json")
        with open(edges_file, 'w') as f:
            json.dump(edges, f, indent=2)
        output_files.append(nodes_file)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, "glean_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        output_files.append(metrics_file)
        
        # Save column patterns
        patterns_file = os.path.join(output_dir, "glean_column_patterns.json")
        with open(patterns_file, 'w') as f:
            json.dump(column_patterns, f, indent=2)
        output_files.append(patterns_file)
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "glean_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(result["metadata"], f, indent=2)
        output_files.append(metadata_file)
        
        result["output_files"] = output_files
        logger.info(f"Saved Glean training data to {output_dir}")
    
    logger.info(f"Ingested {len(nodes)} nodes, {len(edges)} edges from Glean")
    
    return result

