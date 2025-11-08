"""Batch Processing for GNN Models.

This module provides efficient batch processing for multiple graphs,
embedding caching, and memory optimization.
"""

import logging
import os
import hashlib
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np

try:
    import torch
    try:
        from torch_geometric.data import Data, Batch
    except ImportError:
        pass
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for GNN embeddings to avoid recomputation."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size: int = 1000
    ):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size: Maximum number of cached embeddings
        """
        self.cache_dir = cache_dir or os.getenv("GNN_CACHE_DIR", "./gnn_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_size = max_size
        self.cache_index = {}
        self._load_cache_index()
    
    def _get_cache_key(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key from graph and config.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            config: Optional configuration dict
        
        Returns:
            Cache key (hash)
        """
        # Create hashable representation
        graph_repr = {
            "nodes": sorted([(n.get("id", ""), n.get("type", "")) for n in nodes]),
            "edges": sorted([(e.get("source_id", ""), e.get("target_id", "")) for e in edges]),
            "config": config or {}
        }
        
        graph_str = json.dumps(graph_repr, sort_keys=True)
        return hashlib.sha256(graph_str.encode()).hexdigest()
    
    def get(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached embeddings.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            config: Optional configuration dict
        
        Returns:
            Cached embeddings or None
        """
        cache_key = self._get_cache_key(nodes, edges, config)
        
        if cache_key in self.cache_index:
            cache_file = self.cache_index[cache_key]
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                # Remove from index
                del self.cache_index[cache_key]
                self._save_cache_index()
        
        return None
    
    def put(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        embeddings: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ):
        """Store embeddings in cache.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            embeddings: Embeddings to cache
            config: Optional configuration dict
        """
        cache_key = self._get_cache_key(nodes, edges, config)
        
        # Check cache size
        if len(self.cache_index) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache_index))
            self._remove_cache_entry(oldest_key)
        
        # Save to cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embeddings, f)
            
            self.cache_index[cache_key] = cache_file
            self._save_cache_index()
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry.
        
        Args:
            cache_key: Cache key to remove
        """
        if cache_key in self.cache_index:
            cache_file = self.cache_index[cache_key]
            try:
                os.remove(cache_file)
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")
            
            del self.cache_index[cache_key]
            self._save_cache_index()
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, "r") as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        try:
            with open(index_file, "w") as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def clear(self):
        """Clear all cached embeddings."""
        for cache_key in list(self.cache_index.keys()):
            self._remove_cache_entry(cache_key)
        self.cache_index = {}
        self._save_cache_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_size = 0
        for cache_file in self.cache_index.values():
            if os.path.exists(cache_file):
                total_size += os.path.getsize(cache_file)
        
        return {
            "num_entries": len(self.cache_index),
            "max_size": self.max_size,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }


class GraphBatchProcessor:
    """Batch processor for multiple graphs."""
    
    def __init__(
        self,
        embedder=None,
        cache: Optional[EmbeddingCache] = None,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """Initialize batch processor.
        
        Args:
            embedder: GNN embedder instance
            cache: Embedding cache (optional)
            batch_size: Batch size for processing
            device: Device to use
        """
        self.embedder = embedder
        self.cache = cache
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def process_graphs_batch(
        self,
        graphs: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple graphs in batch.
        
        Args:
            graphs: List of graphs, each with 'nodes' and 'edges'
            use_cache: Whether to use embedding cache
        
        Returns:
            List of embedding results
        """
        results = []
        
        for i in range(0, len(graphs), self.batch_size):
            batch_graphs = graphs[i:i + self.batch_size]
            batch_results = self._process_batch(batch_graphs, use_cache)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(
        self,
        graphs: List[Dict[str, Any]],
        use_cache: bool
    ) -> List[Dict[str, Any]]:
        """Process a batch of graphs.
        
        Args:
            graphs: Batch of graphs
            use_cache: Whether to use cache
        
        Returns:
            List of results
        """
        results = []
        
        for graph in graphs:
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            config = graph.get("config", {})
            
            # Check cache
            if use_cache and self.cache:
                cached = self.cache.get(nodes, edges, config)
                if cached is not None:
                    results.append(cached)
                    continue
            
            # Generate embeddings
            if self.embedder:
                try:
                    embeddings = self.embedder.generate_embeddings(
                        nodes, edges,
                        graph_level=True
                    )
                    
                    # Cache if enabled
                    if use_cache and self.cache and "error" not in embeddings:
                        self.cache.put(nodes, edges, embeddings, config)
                    
                    results.append(embeddings)
                except Exception as e:
                    logger.error(f"Failed to process graph: {e}")
                    results.append({"error": str(e)})
            else:
                results.append({"error": "Embedder not available"})
        
        return results
    
    def process_graphs_parallel(
        self,
        graphs: List[Dict[str, Any]],
        num_workers: int = 4,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Process graphs in parallel (using multiprocessing).
        
        Args:
            graphs: List of graphs
            num_workers: Number of worker processes
            use_cache: Whether to use cache
        
        Returns:
            List of results
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        results = [None] * len(graphs)
        
        # Process in chunks
        chunk_size = max(1, len(graphs) // num_workers)
        chunks = [
            graphs[i:i + chunk_size]
            for i in range(0, len(graphs), chunk_size)
        ]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for chunk_idx, chunk in enumerate(chunks):
                start_idx = chunk_idx * chunk_size
                future = executor.submit(self._process_chunk, chunk, use_cache)
                futures.append((start_idx, future))
            
            for start_idx, future in futures:
                chunk_results = future.result()
                for i, result in enumerate(chunk_results):
                    results[start_idx + i] = result
        
        return results
    
    def _process_chunk(
        self,
        graphs: List[Dict[str, Any]],
        use_cache: bool
    ) -> List[Dict[str, Any]]:
        """Process a chunk of graphs (for multiprocessing).
        
        Args:
            graphs: Chunk of graphs
            use_cache: Whether to use cache
        
        Returns:
            List of results
        """
        return self._process_batch(graphs, use_cache)


class MemoryOptimizer:
    """Memory optimization utilities for GNN processing."""
    
    @staticmethod
    def optimize_graph_data(
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        remove_unused_properties: bool = True,
        compress_ids: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Optimize graph data for memory efficiency.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            remove_unused_properties: Remove unused node/edge properties
            compress_ids: Use integer IDs instead of strings
        
        Returns:
            Optimized (nodes, edges)
        """
        optimized_nodes = []
        optimized_edges = []
        
        # Create ID mapping if compressing
        id_mapping = {}
        if compress_ids:
            for i, node in enumerate(nodes):
                node_id = node.get("id", "")
                if node_id:
                    id_mapping[node_id] = i
        
        # Optimize nodes
        for node in nodes:
            optimized_node = {
                "id": id_mapping.get(node.get("id", ""), node.get("id", "")) if compress_ids else node.get("id", ""),
                "type": node.get("type", "unknown")
            }
            
            # Keep only essential properties
            if not remove_unused_properties:
                optimized_node.update(node)
            
            optimized_nodes.append(optimized_node)
        
        # Optimize edges
        for edge in edges:
            source_id = edge.get("source_id", "")
            target_id = edge.get("target_id", "")
            
            if compress_ids:
                source_id = id_mapping.get(source_id, source_id)
                target_id = id_mapping.get(target_id, target_id)
            
            optimized_edge = {
                "source_id": source_id,
                "target_id": target_id,
                "label": edge.get("label", "")
            }
            
            if not remove_unused_properties:
                optimized_edge.update(edge)
            
            optimized_edges.append(optimized_edge)
        
        return optimized_nodes, optimized_edges
    
    @staticmethod
    def estimate_memory_usage(
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Estimate memory usage of graph data.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
        
        Returns:
            Dictionary with memory estimates (in MB)
        """
        import sys
        
        # Estimate node memory
        node_size = sys.getsizeof(nodes)
        for node in nodes:
            node_size += sum(sys.getsizeof(v) for v in node.values())
        
        # Estimate edge memory
        edge_size = sys.getsizeof(edges)
        for edge in edges:
            edge_size += sum(sys.getsizeof(v) for v in edge.values())
        
        total_size = node_size + edge_size
        
        return {
            "nodes_mb": node_size / (1024 * 1024),
            "edges_mb": edge_size / (1024 * 1024),
            "total_mb": total_size / (1024 * 1024),
            "num_nodes": len(nodes),
            "num_edges": len(edges)
        }
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache if available."""
        if HAS_PYG and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

