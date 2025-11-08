"""Anomaly detection-specific LangGraph workflow."""

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


class AnomalyState(TypedDict):
    """State for anomaly detection workflow."""
    query: str
    storyline_id: str
    gnn_anomalies: Dict[str, Any]
    enriched_report: str
    final_response: str


class AnomalyWorkflow:
    """Workflow specifically for anomaly detection."""
    
    def __init__(self, narrative_gnn, narrative_graph, bridge):
        """Initialize anomaly workflow."""
        self.narrative_gnn = narrative_gnn
        self.narrative_graph = narrative_graph
        self.bridge = bridge
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build anomaly workflow graph."""
        workflow = StateGraph(AnomalyState)
        
        workflow.add_node("identify_storyline", self.identify_storyline)
        workflow.add_node("detect_anomalies", self.detect_anomalies)
        workflow.add_node("enrich_report", self.enrich_report)
        
        workflow.set_entry_point("identify_storyline")
        workflow.add_edge("identify_storyline", "detect_anomalies")
        workflow.add_edge("detect_anomalies", "enrich_report")
        workflow.add_edge("enrich_report", END)
        
        return workflow.compile()
    
    def identify_storyline(self, state: AnomalyState) -> Dict[str, Any]:
        """Identify storyline from query."""
        query = state.get("query", "")
        params = self.bridge.langgraph_to_gnn(query)
        return {"storyline_id": params.get("storyline_id")}
    
    def detect_anomalies(self, state: AnomalyState) -> Dict[str, Any]:
        """Detect anomalies using GNN."""
        storyline_id = state.get("storyline_id")
        result = self.narrative_gnn.forward(
            self.narrative_graph,
            current_time=0.0,
            task_mode="detect_anomalies",
            storyline_id=storyline_id
        )
        return {"gnn_anomalies": result}
    
    def enrich_report(self, state: AnomalyState) -> Dict[str, Any]:
        """Enrich anomaly report with LLM."""
        gnn_output = state.get("gnn_anomalies", {})
        enriched = self.bridge.gnn_to_langgraph(gnn_output, "detect_anomalies")
        return {
            "enriched_report": enriched.get("enriched_report", ""),
            "final_response": enriched.get("enriched_report", "")
        }

