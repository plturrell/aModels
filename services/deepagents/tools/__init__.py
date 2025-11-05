"""Custom tools for DeepAgents integration with aModels services."""

from .knowledge_graph_tool import query_knowledge_graph
from .agentflow_tool import run_agentflow_flow
from .orchestration_tool import run_orchestration_chain
from .gpu_tool import allocate_gpu, release_gpu, query_gpu_status, analyze_workload

__all__ = [
    "query_knowledge_graph",
    "run_agentflow_flow",
    "run_orchestration_chain",
    "allocate_gpu",
    "release_gpu",
    "query_gpu_status",
    "analyze_workload",
]

