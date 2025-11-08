"""Prediction-specific LangGraph workflow."""

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


class PredictionState(TypedDict):
    """State for prediction workflow."""
    query: str
    time_point: float
    storyline_id: str
    gnn_prediction: Dict[str, Any]
    enriched_prediction: str
    final_response: str


class PredictionWorkflow:
    """Workflow specifically for prediction generation."""
    
    def __init__(self, narrative_gnn, narrative_graph, bridge):
        """Initialize prediction workflow."""
        self.narrative_gnn = narrative_gnn
        self.narrative_graph = narrative_graph
        self.bridge = bridge
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build prediction workflow graph."""
        workflow = StateGraph(PredictionState)
        
        workflow.add_node("extract_time", self.extract_time)
        workflow.add_node("get_prediction", self.get_prediction)
        workflow.add_node("enrich_prediction", self.enrich_prediction)
        
        workflow.set_entry_point("extract_time")
        workflow.add_edge("extract_time", "get_prediction")
        workflow.add_edge("get_prediction", "enrich_prediction")
        workflow.add_edge("enrich_prediction", END)
        
        return workflow.compile()
    
    def extract_time(self, state: PredictionState) -> Dict[str, Any]:
        """Extract time point from query."""
        query = state.get("query", "")
        params = self.bridge.langgraph_to_gnn(query)
        return {
            "time_point": params.get("time_point", 0.0),
            "storyline_id": params.get("storyline_id")
        }
    
    def get_prediction(self, state: PredictionState) -> Dict[str, Any]:
        """Get prediction from GNN."""
        time_point = state.get("time_point", 0.0)
        storyline_id = state.get("storyline_id")
        result = self.narrative_gnn.forward(
            self.narrative_graph,
            current_time=time_point,
            task_mode="predict",
            storyline_id=storyline_id
        )
        return {"gnn_prediction": result}
    
    def enrich_prediction(self, state: PredictionState) -> Dict[str, Any]:
        """Enrich prediction with LLM."""
        gnn_output = state.get("gnn_prediction", {})
        enriched = self.bridge.gnn_to_langgraph(gnn_output, "predict")
        return {
            "enriched_prediction": enriched.get("enriched_prediction", ""),
            "final_response": enriched.get("enriched_prediction", "")
        }

