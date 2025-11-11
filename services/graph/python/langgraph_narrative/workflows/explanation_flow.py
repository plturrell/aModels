"""Explanation-specific LangGraph workflow."""

import logging
from typing import TypedDict, Dict, Any

try:
    from langgraph.graph import StateGraph, END, START
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = None
    END = None
    START = None

logger = logging.getLogger(__name__)


class ExplanationState(TypedDict):
    """State for explanation workflow."""
    query: str
    entities: list
    storyline_id: str
    gnn_explanation: Dict[str, Any]
    enriched_explanation: str
    final_response: str


class ExplanationWorkflow:
    """Workflow specifically for explanation generation."""
    
    def __init__(self, narrative_gnn, narrative_graph, bridge):
        """Initialize explanation workflow.
        
        Args:
            narrative_gnn: MultiPurposeNarrativeGNN instance
            narrative_graph: NarrativeGraph instance
            bridge: LangGraphNarrativeBridge instance
        """
        self.narrative_gnn = narrative_gnn
        self.narrative_graph = narrative_graph
        self.bridge = bridge
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build explanation workflow graph."""
        workflow = StateGraph(ExplanationState)
        
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("get_explanation", self.get_explanation)
        workflow.add_node("enrich_explanation", self.enrich_explanation)
        
        workflow.set_entry_point("extract_entities")
        workflow.add_edge("extract_entities", "get_explanation")
        workflow.add_edge("get_explanation", "enrich_explanation")
        workflow.add_edge("enrich_explanation", END)
        
        return workflow.compile()
    
    def extract_entities(self, state: ExplanationState) -> Dict[str, Any]:
        """Extract entities from query."""
        query = state.get("query", "")
        params = self.bridge.langgraph_to_gnn(query)
        return {
            "entities": params.get("entities", []),
            "storyline_id": params.get("storyline_id")
        }
    
    def get_explanation(self, state: ExplanationState) -> Dict[str, Any]:
        """Get explanation from GNN."""
        storyline_id = state.get("storyline_id")
        result = self.narrative_gnn.forward(
            self.narrative_graph,
            current_time=0.0,
            task_mode="explain",
            storyline_id=storyline_id
        )
        return {"gnn_explanation": result}
    
    def enrich_explanation(self, state: ExplanationState) -> Dict[str, Any]:
        """Enrich explanation with LLM."""
        gnn_output = state.get("gnn_explanation", {})
        enriched = self.bridge.gnn_to_langgraph(gnn_output, "explain")
        return {
            "enriched_explanation": enriched.get("enriched_explanation", ""),
            "final_response": enriched.get("enriched_explanation", "")
        }

