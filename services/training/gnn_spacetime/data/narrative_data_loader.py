"""Data loader for converting raw events into narrative graphs."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from ..narrative.narrative_graph import NarrativeGraph
from ..narrative.narrative_node import NarrativeNode
from ..narrative.narrative_edge import NarrativeEdge
from ..narrative.storyline import Storyline, NarrativeType

logger = logging.getLogger(__name__)


class NarrativeDataLoader:
    """Loads and converts raw event streams into narrative graphs."""
    
    def __init__(self):
        """Initialize narrative data loader."""
        logger.info("Initialized NarrativeDataLoader")
    
    def convert_raw_events_to_narrative_graph(
        self,
        raw_events: List[Dict[str, Any]],
        story_themes: Optional[List[Dict[str, Any]]] = None,
        time_normalization: bool = True
    ) -> NarrativeGraph:
        """Transform temporal event streams into narrative graphs.
        
        Args:
            raw_events: List of event dicts with at least:
                - "time": timestamp or time value
                - "source": source entity ID
                - "target": target entity ID (optional)
                - "event_type": type of event
                - "description": event description
            story_themes: Optional list of story theme dicts with:
                - "theme_id": unique identifier
                - "theme": description
                - "narrative_type": "explanation", "prediction", or "anomaly_detection"
            time_normalization: If True, normalize times to start from 0.0
            
        Returns:
            NarrativeGraph instance
        """
        logger.info(f"Converting {len(raw_events)} raw events to narrative graph")
        
        # Normalize times if requested
        if time_normalization and raw_events:
            times = [e.get("time", 0.0) for e in raw_events]
            min_time = min(times) if times else 0.0
            if min_time > 0:
                raw_events = [
                    {**e, "time": e.get("time", 0.0) - min_time}
                    for e in raw_events
                ]
        
        # Extract unique entities
        entities = self._extract_entities(raw_events)
        
        # Create narrative nodes
        nodes = []
        for entity_id, entity_data in entities.items():
            node = self._create_narrative_node(entity_id, entity_data, raw_events)
            nodes.append(node)
        
        # Create narrative edges from events
        edges = []
        for event in raw_events:
            if "source" in event and "target" in event:
                edge = self._create_narrative_edge(event, raw_events)
                if edge:
                    edges.append(edge)
        
        # Create storylines from themes
        storylines = {}
        if story_themes:
            for theme_data in story_themes:
                storyline = self._create_storyline(theme_data, raw_events, nodes, edges)
                if storyline:
                    storylines[storyline.storyline_id] = storyline
        
        # If no themes provided, create default storyline
        if not storylines:
            default_storyline = self._create_default_storyline(raw_events, nodes, edges)
            if default_storyline:
                storylines[default_storyline.storyline_id] = default_storyline
        
        # Create narrative graph
        graph = NarrativeGraph(nodes=nodes, edges=edges, storylines=storylines)
        
        logger.info(
            f"Created narrative graph: {len(nodes)} nodes, {len(edges)} edges, "
            f"{len(storylines)} storylines"
        )
        
        return graph
    
    def _extract_entities(self, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Extract unique entities from events.
        
        Args:
            events: List of event dicts
            
        Returns:
            Dict mapping entity_id -> entity_data
        """
        entities = {}
        
        for event in events:
            # Extract source entity
            if "source" in event:
                source_id = event["source"]
                if source_id not in entities:
                    entities[source_id] = {
                        "id": source_id,
                        "type": event.get("source_type", "entity"),
                        "first_seen": event.get("time", 0.0),
                        "events": []
                    }
                entities[source_id]["events"].append(event)
            
            # Extract target entity
            if "target" in event:
                target_id = event["target"]
                if target_id not in entities:
                    entities[target_id] = {
                        "id": target_id,
                        "type": event.get("target_type", "entity"),
                        "first_seen": event.get("time", 0.0),
                        "events": []
                    }
                entities[target_id]["events"].append(event)
        
        return entities
    
    def _create_narrative_node(
        self,
        entity_id: str,
        entity_data: Dict[str, Any],
        all_events: List[Dict[str, Any]]
    ) -> NarrativeNode:
        """Create narrative node from entity data.
        
        Args:
            entity_id: Entity identifier
            entity_data: Entity metadata
            all_events: All events for context
            
        Returns:
            NarrativeNode instance
        """
        # Create static embedding (placeholder - in practice would use actual embeddings)
        static_emb = None
        if HAS_TORCH:
            static_emb = torch.randn(128)  # Placeholder
        
        # Determine lifespan
        entity_events = entity_data.get("events", [])
        if entity_events:
            times = [e.get("time", 0.0) for e in entity_events]
            start_time = min(times)
            end_time = max(times) if len(set(times)) > 1 else None
            lifespan = (start_time, end_time)
        else:
            lifespan = (entity_data.get("first_seen", 0.0), None)
        
        # Create state history from events
        state_history = []
        for event in entity_events:
            time = event.get("time", 0.0)
            # Create simple state vector from event
            if HAS_TORCH:
                state = torch.randn(64)  # Placeholder
                state_history.append((time, state))
        
        return NarrativeNode(
            node_id=entity_id,
            node_type=entity_data.get("type", "entity"),
            static_embedding=static_emb,
            lifespan=lifespan,
            state_history=state_history,
            properties={"num_events": len(entity_events)}
        )
    
    def _create_narrative_edge(
        self,
        event: Dict[str, Any],
        all_events: List[Dict[str, Any]]
    ) -> Optional[NarrativeEdge]:
        """Create narrative edge from event.
        
        Args:
            event: Event dict
            all_events: All events for context
            
        Returns:
            NarrativeEdge instance or None
        """
        source_id = event.get("source")
        target_id = event.get("target")
        
        if not source_id or not target_id:
            return None
        
        # Determine temporal scope
        event_time = event.get("time", 0.0)
        temporal_scope = (event_time, None)  # Assume ongoing unless specified
        
        # Create relation embedding (placeholder)
        relation_emb = None
        if HAS_TORCH:
            relation_emb = torch.randn(64)  # Placeholder
        
        return NarrativeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=event.get("event_type", "relates_to"),
            relation_embedding=relation_emb,
            temporal_scope=temporal_scope,
            base_weight=1.0
        )
    
    def _create_storyline(
        self,
        theme_data: Dict[str, Any],
        events: List[Dict[str, Any]],
        nodes: List[NarrativeNode],
        edges: List[NarrativeEdge]
    ) -> Optional[Storyline]:
        """Create storyline from theme data.
        
        Args:
            theme_data: Theme metadata
            events: All events
            nodes: All nodes
            edges: All edges
            
        Returns:
            Storyline instance or None
        """
        theme_id = theme_data.get("theme_id", "default_story")
        theme = theme_data.get("theme", "Untitled story")
        narrative_type_str = theme_data.get("narrative_type", "general")
        
        # Map string to enum
        narrative_type_map = {
            "explanation": NarrativeType.EXPLANATION,
            "prediction": NarrativeType.PREDICTION,
            "anomaly_detection": NarrativeType.ANOMALY_DETECTION,
            "general": NarrativeType.GENERAL
        }
        narrative_type = narrative_type_map.get(narrative_type_str, NarrativeType.GENERAL)
        
        # Extract causal links from events
        causal_links = []
        for event in events:
            if "source" in event and "target" in event:
                causal_links.append((
                    event["source"],
                    event["target"],
                    event.get("event_type", "relates_to")
                ))
        
        # Extract key events
        key_events = [
            {
                "time": e.get("time", 0.0),
                "node_id": e.get("source"),
                "description": e.get("description", "Event")
            }
            for e in events[:10]  # Top 10 events
        ]
        
        return Storyline(
            storyline_id=theme_id,
            theme=theme,
            narrative_type=narrative_type,
            causal_links=causal_links,
            key_events=key_events
        )
    
    def _create_default_storyline(
        self,
        events: List[Dict[str, Any]],
        nodes: List[NarrativeNode],
        edges: List[NarrativeEdge]
    ) -> Optional[Storyline]:
        """Create default storyline from events.
        
        Args:
            events: All events
            nodes: All nodes
            edges: All edges
            
        Returns:
            Storyline instance or None
        """
        if not events:
            return None
        
        return Storyline(
            storyline_id="default_story",
            theme="Default narrative from event stream",
            narrative_type=NarrativeType.GENERAL,
            causal_links=[
                (e.get("source"), e.get("target"), e.get("event_type", "relates_to"))
                for e in events if "source" in e and "target" in e
            ],
            key_events=[
                {
                    "time": e.get("time", 0.0),
                    "node_id": e.get("source"),
                    "description": e.get("description", "Event")
                }
                for e in events[:10]
            ]
        )


def load_narrative_from_temporal_graph(temporal_graph) -> NarrativeGraph:
    """Convert existing TemporalGraph to NarrativeGraph (backward compatibility).
    
    Args:
        temporal_graph: TemporalGraph instance
        
    Returns:
        NarrativeGraph instance
    """
    from ..data.temporal_graph import TemporalGraph
    from ..data.temporal_node import TemporalNode
    from ..data.temporal_edge import TemporalEdge
    
    # Convert nodes
    narrative_nodes = []
    for node in temporal_graph.nodes.values():
        if isinstance(node, TemporalNode):
            narrative_node = NarrativeNode(
                node_id=node.node_id,
                node_type=node.node_type,
                features=node.features,
                static_embedding=node.static_embedding,
                lifespan=node.lifespan,
                state_history=node.state_history,
                properties=node.properties
            )
            narrative_nodes.append(narrative_node)
    
    # Convert edges
    narrative_edges = []
    for edge in temporal_graph.edges:
        if isinstance(edge, TemporalEdge):
            narrative_edge = NarrativeEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation_type=edge.relation_type,
                relation_embedding=edge.relation_embedding,
                temporal_scope=edge.temporal_scope,
                weight_function=edge.weight_function,
                base_weight=edge.base_weight,
                properties=edge.properties
            )
            narrative_edges.append(narrative_edge)
    
    # Create default storyline
    default_storyline = Storyline(
        storyline_id="converted_story",
        theme="Converted from temporal graph",
        narrative_type=NarrativeType.GENERAL
    )
    
    return NarrativeGraph(
        nodes=narrative_nodes,
        edges=narrative_edges,
        storylines={"converted_story": default_storyline}
    )

