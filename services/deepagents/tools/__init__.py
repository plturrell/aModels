"""Custom tools for DeepAgents integration with aModels services."""

from .knowledge_graph_tool import query_knowledge_graph
from .agentflow_tool import run_agentflow_flow
from .orchestration_tool import run_orchestration_chain
from .gpu_tool import allocate_gpu, release_gpu, query_gpu_status, analyze_workload
from .signavio_tool import signavio_stub_upload, signavio_stub_fetch_view

# GNN tools (Priority 2)
try:
    from .gnn_tool import (
        query_gnn_embeddings,
        query_gnn_structural_insights,
        predict_relationships_gnn,
        classify_nodes_gnn,
        query_domain_gnn
    )
    HAS_GNN_TOOLS = True
except ImportError:
    HAS_GNN_TOOLS = False
    query_gnn_embeddings = None
    query_gnn_structural_insights = None
    predict_relationships_gnn = None
    classify_nodes_gnn = None
    query_domain_gnn = None

# Query router (Priority 3)
try:
    from .query_router import hybrid_query, route_query
    HAS_QUERY_ROUTER = True
except ImportError:
    HAS_QUERY_ROUTER = False
    hybrid_query = None
    route_query = None

# MCTS tools (LNN/MCTS integration)
try:
    from .mcts_tool import (
        plan_narrative_path_mcts,
        what_if_analysis_mcts,
        reflective_mcts_debate
    )
    HAS_MCTS_TOOLS = True
except ImportError:
    HAS_MCTS_TOOLS = False
    plan_narrative_path_mcts = None
    what_if_analysis_mcts = None
    reflective_mcts_debate = None

__all__ = [
    "query_knowledge_graph",
    "run_agentflow_flow",
    "run_orchestration_chain",
    "allocate_gpu",
    "release_gpu",
    "query_gpu_status",
    "analyze_workload",
    "signavio_stub_upload",
    "signavio_stub_fetch_view",
]

# Add GNN tools if available
if HAS_GNN_TOOLS:
    __all__.extend([
        "query_gnn_embeddings",
        "query_gnn_structural_insights",
        "predict_relationships_gnn",
        "classify_nodes_gnn",
        "query_domain_gnn",
    ])

# Add query router tools if available
if HAS_QUERY_ROUTER:
    __all__.extend([
        "hybrid_query",
        "route_query",
    ])

# Add MCTS tools if available
if HAS_MCTS_TOOLS:
    __all__.extend([
        "plan_narrative_path_mcts",
        "what_if_analysis_mcts",
        "reflective_mcts_debate",
    ])

