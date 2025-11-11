"""
Unified graph data models for Python services.

This module provides Pydantic models that match the Go GraphData schema,
ensuring bidirectional compatibility between Go and Python services.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class Node(BaseModel):
    """Unified node representation."""
    id: str = Field(..., description="Node identifier")
    type: str = Field(..., description="Node type")
    label: Optional[str] = Field(None, description="Node label")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Node properties")


class Edge(BaseModel):
    """Unified edge representation."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: Optional[str] = Field(None, description="Edge label/relation type")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Edge properties")


class Metadata(BaseModel):
    """Graph-level metadata."""
    project_id: Optional[str] = Field(None, description="Project identifier")
    system_id: Optional[str] = Field(None, description="System identifier")
    information_system_id: Optional[str] = Field(None, description="Information system identifier")
    root_node_id: Optional[str] = Field(None, description="Root node identifier")
    metadata_entropy: Optional[float] = Field(None, description="Metadata entropy score")
    kl_divergence: Optional[float] = Field(None, description="KL divergence score")
    warnings: Optional[List[str]] = Field(default_factory=list, description="Processing warnings")
    additional: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class Quality(BaseModel):
    """Data quality metrics."""
    score: float = Field(..., description="Quality score (0-1)")
    level: str = Field(..., description="Quality level (excellent, good, fair, poor, critical)")
    issues: Optional[List[str]] = Field(default_factory=list, description="Quality issues")
    recommendations: Optional[List[str]] = Field(default_factory=list, description="Quality recommendations")
    processing_strategy: Optional[str] = Field(None, description="Recommended processing strategy")


class GraphData(BaseModel):
    """Unified graph data format."""
    nodes: List[Node] = Field(..., description="Graph nodes")
    edges: List[Edge] = Field(..., description="Graph edges")
    metadata: Optional[Metadata] = Field(None, description="Graph metadata")
    quality: Optional[Quality] = Field(None, description="Data quality metrics")

    def to_gnn_format(self) -> Dict[str, Any]:
        """Convert to GNN service format (nodes/edges arrays)."""
        return {
            "nodes": [node.dict() for node in self.nodes],
            "edges": [edge.dict() for edge in self.edges],
        }

    def to_neo4j_format(self) -> Dict[str, Any]:
        """Convert to Neo4j-compatible format."""
        result = {
            "nodes": [node.dict() for node in self.nodes],
            "edges": [edge.dict() for edge in self.edges],
        }
        
        if self.metadata:
            metadata_dict = self.metadata.dict(exclude_none=True)
            if metadata_dict:
                result["metadata"] = metadata_dict
        
        if self.quality:
            result["quality"] = self.quality.dict(exclude_none=True)
        
        return result

    def validate_graph(self) -> List[str]:
        """Validate graph structure and return list of issues."""
        issues = []
        
        # Check for duplicate node IDs
        node_ids = set()
        for node in self.nodes:
            if not node.id:
                issues.append("Node with empty ID found")
            elif node.id in node_ids:
                issues.append(f"Duplicate node ID: {node.id}")
            else:
                node_ids.add(node.id)
        
        # Check edges reference valid nodes
        for edge in self.edges:
            if not edge.source or not edge.target:
                issues.append("Edge with empty source or target ID")
            elif edge.source not in node_ids:
                issues.append(f"Edge references non-existent source node: {edge.source}")
            elif edge.target not in node_ids:
                issues.append(f"Edge references non-existent target node: {edge.target}")
        
        return issues


def from_neo4j(neo4j_result: Dict[str, Any]) -> GraphData:
    """
    Convert Neo4j Cypher query result to unified GraphData format.
    
    Args:
        neo4j_result: Neo4j query result in format {"columns": [...], "data": [[...], ...]}
                     or {"nodes": [...], "edges": [...]}
    
    Returns:
        GraphData instance
    """
    nodes = []
    edges = []
    metadata = None
    quality = None
    
    # Handle Neo4j Cypher result format
    if "columns" in neo4j_result and "data" in neo4j_result:
        columns = neo4j_result["columns"]
        data = neo4j_result["data"]
        
        for row in data:
            for i, col in enumerate(columns):
                if i >= len(row):
                    continue
                value = row[i]
                
                # Check if value is a node or relationship
                if isinstance(value, dict):
                    # Check for Neo4j node format
                    if "labels" in value:
                        # This is a Neo4j node
                        node_id = str(value.get("id", value.get("element_id", "")))
                        labels = value.get("labels", [])
                        node_type = labels[0] if labels else "unknown"
                        properties = value.get("properties", {})
                        
                        nodes.append(Node(
                            id=node_id,
                            type=node_type,
                            label=node_id,
                            properties=properties
                        ))
                    elif "startNodeElementId" in value:
                        # This is a Neo4j relationship
                        source_id = value.get("startNodeElementId", "")
                        target_id = value.get("endNodeElementId", "")
                        rel_type = value.get("type", "")
                        properties = value.get("properties", {})
                        
                        edges.append(Edge(
                            source=source_id,
                            target=target_id,
                            label=rel_type,
                            properties=properties
                        ))
    
    # Handle direct nodes/edges format
    if "nodes" in neo4j_result:
        nodes_data = neo4j_result["nodes"]
        if isinstance(nodes_data, list):
            for node_data in nodes_data:
                if isinstance(node_data, dict):
                    nodes.append(Node(
                        id=node_data.get("id", node_data.get("node_id", "")),
                        type=node_data.get("type", node_data.get("node_type", "unknown")),
                        label=node_data.get("label"),
                        properties=node_data.get("properties", {})
                    ))
    
    if "edges" in neo4j_result:
        edges_data = neo4j_result["edges"]
        if isinstance(edges_data, list):
            for edge_data in edges_data:
                if isinstance(edge_data, dict):
                    edges.append(Edge(
                        source=edge_data.get("source", edge_data.get("source_id", "")),
                        target=edge_data.get("target", edge_data.get("target_id", "")),
                        label=edge_data.get("label", edge_data.get("relation_type", "")),
                        properties=edge_data.get("properties", {})
                    ))
    
    # Extract metadata
    if "metadata" in neo4j_result:
        metadata_dict = neo4j_result["metadata"]
        if isinstance(metadata_dict, dict):
            metadata = Metadata(
                project_id=metadata_dict.get("project_id"),
                system_id=metadata_dict.get("system_id"),
                information_system_id=metadata_dict.get("information_system_id"),
                root_node_id=metadata_dict.get("root_node_id"),
                metadata_entropy=metadata_dict.get("metadata_entropy"),
                kl_divergence=metadata_dict.get("kl_divergence"),
                warnings=metadata_dict.get("warnings", [])
            )
    
    # Extract quality
    if "quality" in neo4j_result:
        quality_dict = neo4j_result["quality"]
        if isinstance(quality_dict, dict):
            quality = Quality(
                score=quality_dict.get("score", 0.0),
                level=quality_dict.get("level", "unknown"),
                issues=quality_dict.get("issues", []),
                recommendations=quality_dict.get("recommendations", []),
                processing_strategy=quality_dict.get("processing_strategy")
            )
    
    return GraphData(
        nodes=nodes,
        edges=edges,
        metadata=metadata,
        quality=quality
    )


def to_gnn_format(graph_data: GraphData) -> Dict[str, Any]:
    """
    Convert GraphData to GNN service format.
    
    Args:
        graph_data: GraphData instance
    
    Returns:
        Dictionary with "nodes" and "edges" keys
    """
    return graph_data.to_gnn_format()


def from_gnn(gnn_response: Dict[str, Any]) -> GraphData:
    """
    Convert GNN service response to unified GraphData format.
    
    Args:
        gnn_response: GNN service response
    
    Returns:
        GraphData instance
    """
    nodes = []
    edges = []
    
    # Extract nodes
    if "nodes" in gnn_response:
        nodes_data = gnn_response["nodes"]
        if isinstance(nodes_data, list):
            for node_data in nodes_data:
                if isinstance(node_data, dict):
                    nodes.append(Node(
                        id=node_data.get("id", node_data.get("node_id", "")),
                        type=node_data.get("type", node_data.get("node_type", "unknown")),
                        label=node_data.get("label"),
                        properties=node_data.get("properties", {})
                    ))
    elif "node_embeddings" in gnn_response:
        # Handle node embeddings format
        for node_id in gnn_response["node_embeddings"]:
            nodes.append(Node(
                id=str(node_id),
                type="unknown"
            ))
    
    # Extract edges
    if "edges" in gnn_response:
        edges_data = gnn_response["edges"]
        if isinstance(edges_data, list):
            for edge_data in edges_data:
                if isinstance(edge_data, dict):
                    edges.append(Edge(
                        source=edge_data.get("source", edge_data.get("source_id", "")),
                        target=edge_data.get("target", edge_data.get("target_id", "")),
                        label=edge_data.get("label", edge_data.get("relation_type", "")),
                        properties=edge_data.get("properties", {})
                    ))
    
    # Extract metadata if present
    metadata = None
    if "metadata" in gnn_response:
        metadata_dict = gnn_response["metadata"]
        if isinstance(metadata_dict, dict):
            metadata = Metadata(
                project_id=metadata_dict.get("project_id"),
                system_id=metadata_dict.get("system_id"),
                information_system_id=metadata_dict.get("information_system_id"),
                root_node_id=metadata_dict.get("root_node_id"),
                metadata_entropy=metadata_dict.get("metadata_entropy"),
                kl_divergence=metadata_dict.get("kl_divergence"),
                warnings=metadata_dict.get("warnings", [])
            )
    
    return GraphData(
        nodes=nodes,
        edges=edges,
        metadata=metadata
    )

