"""Conversational memory for narrative intelligence.

Tracks conversation history and extracts narrative elements for graph updates.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    init_chat_model = None
    HumanMessage = None

# Import narrative components
import sys
from pathlib import Path
# Try multiple paths for narrative GNN
possible_paths = [
    Path(__file__).parent.parent.parent.parent / "training" / "gnn_spacetime",
    Path(__file__).parent.parent.parent.parent.parent / "services" / "training" / "gnn_spacetime",
]
for path in possible_paths:
    if path.exists():
        sys.path.insert(0, str(path.parent.parent))
        break

try:
    from gnn_spacetime.narrative import (
        NarrativeGraph, NarrativeNode, NarrativeEdge, Storyline, NarrativeType
    )
    HAS_NARRATIVE = True
except ImportError:
    HAS_NARRATIVE = False
    NarrativeGraph = None
    NarrativeNode = None
    NarrativeEdge = None
    Storyline = None
    NarrativeType = None

logger = logging.getLogger(__name__)


class NarrativeConversationMemory:
    """Manages conversational memory and extracts narrative elements."""
    
    def __init__(
        self,
        conversation_graph: Optional[NarrativeGraph] = None,
        llm_model: str = "openai:gpt-4o-mini"
    ):
        """Initialize conversation memory.
        
        Args:
            conversation_graph: Optional narrative graph for conversation tracking
            llm_model: LLM model for entity extraction
        """
        if not HAS_LANGCHAIN:
            raise ImportError("LangChain is required for NarrativeConversationMemory")
        
        self.conversation_graph = conversation_graph
        if not self.conversation_graph and HAS_NARRATIVE:
            # Create empty conversation graph
            self.conversation_graph = NarrativeGraph()
        
        self.query_history = []
        self.llm = init_chat_model(llm_model)
        
        logger.info("Initialized NarrativeConversationMemory")
    
    def update_from_conversation(
        self,
        user_query: str,
        gnn_response: Dict[str, Any],
        llm_response: str
    ):
        """Extract narrative elements from conversation and update graph.
        
        Args:
            user_query: User's query
            gnn_response: GNN response
            llm_response: LLM-generated response
        """
        if not self.conversation_graph:
            return
        
        # Extract entities and relationships
        extracted_entities = self.extract_entities_from_text(
            user_query + " " + llm_response
        )
        
        # Update narrative graph with conversation context
        for entity_id, entity_data in extracted_entities.items():
            if entity_id not in self.conversation_graph.nodes:
                if HAS_NARRATIVE:
                    node = NarrativeNode(
                        node_id=entity_id,
                        node_type=entity_data.get("type", "entity"),
                        narrative_roles={"conversation": {"role": "mentioned"}},
                        explanatory_power=0.5
                    )
                    self.conversation_graph.add_node(node)
        
        # Add conversation as a storyline
        conversation_id = f"conversation_{len(self.query_history)}"
        if HAS_NARRATIVE:
            conversation_storyline = Storyline(
                storyline_id=conversation_id,
                theme=f"User interaction: {user_query[:50]}...",
                narrative_type=NarrativeType.GENERAL
            )
            self.conversation_graph.add_storyline(conversation_storyline)
        
        # Record in history
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "gnn_response": gnn_response,
            "llm_response": llm_response,
            "storyline_id": conversation_id
        })
        
        logger.debug(f"Updated conversation memory with {len(extracted_entities)} entities")
    
    def extract_entities_from_text(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract entities from text using LLM.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dict mapping entity_id -> entity_data
        """
        extraction_prompt = f"""Extract all entities (people, organizations, concepts, systems) mentioned in this text:

Text: {text}

Respond in JSON format:
{{
    "entities": [
        {{
            "id": "entity_id",
            "name": "Entity Name",
            "type": "person|organization|concept|system|other"
        }}
    ]
}}"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            
            # Extract JSON
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                
                # Convert to dict
                entity_dict = {}
                for entity in entities:
                    entity_id = entity.get("id", entity.get("name", "").lower().replace(" ", "_"))
                    entity_dict[entity_id] = {
                        "id": entity_id,
                        "name": entity.get("name", entity_id),
                        "type": entity.get("type", "other")
                    }
                
                return entity_dict
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return {}
    
    def get_conversation_context(self, num_exchanges: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation context.
        
        Args:
            num_exchanges: Number of recent exchanges to return
            
        Returns:
            List of conversation exchanges
        """
        return self.query_history[-num_exchanges:] if self.query_history else []
    
    def find_related_storylines(self, query: str) -> List[str]:
        """Find storylines related to query.
        
        Args:
            query: User query
            
        Returns:
            List of related storyline IDs
        """
        if not self.conversation_graph:
            return []
        
        # Extract keywords from query
        query_lower = query.lower()
        related = []
        
        for storyline_id, storyline in self.conversation_graph.storylines.items():
            theme_lower = storyline.theme.lower()
            # Simple keyword matching
            if any(word in theme_lower for word in query_lower.split() if len(word) > 3):
                related.append(storyline_id)
        
        return related
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history.
        
        Returns:
            Dict with conversation summary
        """
        return {
            "num_exchanges": len(self.query_history),
            "entities_mentioned": len(self.conversation_graph.nodes) if self.conversation_graph else 0,
            "storylines_created": len(self.conversation_graph.storylines) if self.conversation_graph else 0,
            "recent_queries": [
                {
                    "query": h["user_query"][:100],
                    "timestamp": h["timestamp"]
                }
                for h in self.query_history[-5:]
            ]
        }

