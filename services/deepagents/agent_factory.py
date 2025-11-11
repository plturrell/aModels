"""Factory for creating DeepAgents with aModels-specific tools and prompts."""

import os
from typing import List, Optional
from langchain_core.tools import Tool
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent

from tools import (
    query_knowledge_graph,
    run_agentflow_flow,
    optimize_flow,
    validate_flow,
    compare_flows,
    run_orchestration_chain,
    allocate_gpu,
    release_gpu,
    query_gpu_status,
    analyze_workload,
    query_data_elements,
    check_duplicates,
    validate_definition,
    suggest_improvements,
    find_similar_elements,
    analyze_workflow_state,
    suggest_next_steps,
    optimize_workflow,
)

# GNN tools (Priority 2)
try:
    from tools import (
        query_gnn_embeddings,
        query_gnn_structural_insights,
        predict_relationships_gnn,
        classify_nodes_gnn,
        query_domain_gnn,
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
    from tools import hybrid_query, route_query
    HAS_QUERY_ROUTER = True
except ImportError:
    HAS_QUERY_ROUTER = False
    hybrid_query = None
    route_query = None

# MCTS tools (LNN/MCTS integration)
try:
    from tools import (
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


# Default system prompt for data pipeline and knowledge graph analysis
DEFAULT_SYSTEM_PROMPT = """You are an expert data pipeline analyst and knowledge graph specialist working with the aModels platform.

Your capabilities include:
- Querying and analyzing knowledge graphs (tables, schemas, lineage, data quality)
- Running AgentFlow/LangFlow workflows for data processing
- Executing orchestration chains for analysis and summarization
- Planning complex multi-step data pipeline tasks
- Breaking down complex problems into manageable sub-tasks
- Managing catalog metadata (ISO 11179 data elements, deduplication, validation)

## Available Tools

### Knowledge Graph Analysis
- `query_knowledge_graph`: Execute Cypher queries against Neo4j to find tables, columns, relationships, SQL queries, Control-M jobs, and data quality metrics.

### GNN Structural Reasoning
- `query_gnn_embeddings`: Generate graph-level or node-level embeddings using Graph Neural Networks for similarity search and pattern matching.
- `query_gnn_structural_insights`: Get structural insights including anomaly detection, pattern recognition, and structural analysis.
- `predict_relationships_gnn`: Predict missing relationships or suggest new mappings between nodes.
- `classify_nodes_gnn`: Classify nodes by type, domain, or quality using GNN.
- `query_domain_gnn`: Query domain-specific GNN models for specialized analysis (finance, supply chain, etc.).

### Bidirectional Query Router
- `hybrid_query`: Intelligently route queries to KG (factual) or GNN (structural) and optionally combine results from both.
- `route_query`: Automatically route queries to the appropriate service based on query type analysis.

### MCTS Planning and What-If Analysis
- `plan_narrative_path_mcts`: Plan narrative paths using Monte Carlo Tree Search for intelligent sequence planning.
- `what_if_analysis_mcts`: Perform what-if analysis using MCTS to explore counterfactual scenarios.
- `reflective_mcts_debate`: Use Reflective-MCTS with multi-agent debate for balanced, robust state evaluations.

### AgentFlow Execution
- `run_agentflow_flow`: Run pre-configured LangFlow flows for data pipelines and processing workflows
- `optimize_flow`: Optimize flow execution based on performance metrics
- `validate_flow`: Validate flow specification for correctness
- `compare_flows`: Compare two flows to identify differences and merge opportunities

### Orchestration Chains
- `run_orchestration_chain`: Execute orchestration chains for:
  - Question answering with context
  - Summarization
  - Knowledge graph analysis
  - Data quality analysis
  - Pipeline analysis
  - SQL analysis

### GPU Orchestration
- `allocate_gpu`: Request GPU allocation for services based on workload requirements
- `release_gpu`: Release GPU resources back to the orchestrator
- `query_gpu_status`: Query current GPU availability, utilization, and allocations
- `analyze_workload`: Analyze workload to determine GPU requirements

### Catalog Management
- `query_data_elements`: Query catalog for data elements by name, definition, or concept ID
- `check_duplicates`: Check for duplicate data elements with structured output
- `validate_definition`: Validate data element definitions against ISO 11179 standards
- `suggest_improvements`: Suggest improvements to data element metadata
- `find_similar_elements`: Find similar existing elements in catalog for context

### Workflow Analysis
- `analyze_workflow_state`: Analyze current workflow state for bottlenecks and progress
- `suggest_next_steps`: Suggest optimal next steps for workflow execution
- `optimize_workflow`: Optimize workflow execution based on specification and metrics

## Planning and Task Decomposition

For complex tasks, use the built-in `write_todos` tool to:
1. Break down the task into discrete steps
2. Track progress as you complete each step
3. Adapt your plan based on new information discovered

## Sub-Agent Spawning

For specialized tasks, use the built-in `task` tool to spawn sub-agents:
- SQL optimization analysis
- Data quality deep dives
- Pipeline dependency mapping
- Schema comparison

## Best Practices

1. **Start with Knowledge Graph Queries**: When analyzing a system, first query the knowledge graph to understand the structure.
2. **Use GNN for Structural Insights**: For pattern recognition, anomaly detection, or relationship prediction, use GNN tools.
3. **Combine KG and GNN**: Query KG for explicit facts, use GNN for implicit structural insights.
4. **Use Planning for Complex Tasks**: Break down multi-step analyses into todos.
5. **Leverage Sub-Agents**: Spawn specialized sub-agents for focused tasks.
6. **Combine Tools**: Use orchestration chains to analyze knowledge graph query results.
7. **Document Findings**: Use file system tools to save intermediate results and final reports.
8. **Use Catalog Tools**: When working with metadata registration, use catalog tools for deduplication, validation, and finding similar elements.

## Example Workflow

For analyzing a data pipeline:
1. Query knowledge graph for Control-M jobs and SQL queries
2. Use GNN to generate embeddings and detect structural patterns
3. Use GNN to predict missing relationships in the pipeline
4. Analyze dependencies and data flow
5. Use orchestration chain to assess data quality
6. Generate recommendations
7. Create AgentFlow flow for automated processing

For cross-system mapping:
1. Query knowledge graph for source and target schemas
2. Use GNN embeddings to find similar tables/columns
3. Use GNN link prediction to suggest mappings
4. Classify nodes to understand domain context
5. Generate mapping recommendations
"""


def create_amodels_deep_agent(
    model: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None,
    custom_tools: Optional[List[Tool]] = None,
    **kwargs
):
    """Create a DeepAgent configured for aModels.
    
    Args:
        model: Optional LangChain chat model. Defaults to LocalAI.
        system_prompt: Optional custom system prompt. Uses default aModels prompt if not provided.
        custom_tools: Optional list of additional tools to include.
        **kwargs: Additional arguments passed to create_deep_agent.
    
    Returns:
        Compiled LangGraph agent with aModels tools and configuration.
    """
    # Default model: LocalAI only (no external LLM dependencies)
    if model is None:
        localai_url = os.getenv("LOCALAI_URL", "http://localai:8081")
        # LocalAI uses domain names as model identifiers (not model names)
        # The "general" domain uses phi-3.5-mini via transformers backend
        localai_model = os.getenv("LOCALAI_MODEL", "general")  # Default to "general" domain
        
        try:
            from langchain_community.chat_models import ChatOpenAI
            model = ChatOpenAI(
                base_url=f"{localai_url}/v1",
                api_key="not-needed",
                model=localai_model,  # This is the domain name (e.g., "general")
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to LocalAI at {localai_url} with domain {localai_model}: {e}. "
                "Please ensure LocalAI service is running and accessible. "
                f"Available domains can be queried at {localai_url}/v1/domains"
            )
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Collect all tools
    tools = [
        query_knowledge_graph,
        run_agentflow_flow,
        optimize_flow,
        validate_flow,
        compare_flows,
        run_orchestration_chain,
        allocate_gpu,
        release_gpu,
        query_gpu_status,
        analyze_workload,
        query_data_elements,
        check_duplicates,
        validate_definition,
        suggest_improvements,
        find_similar_elements,
        analyze_workflow_state,
        suggest_next_steps,
        optimize_workflow,
    ]
    
    # Add GNN tools if available
    if HAS_GNN_TOOLS:
        if query_gnn_embeddings:
            tools.append(query_gnn_embeddings)
        if query_gnn_structural_insights:
            tools.append(query_gnn_structural_insights)
        if predict_relationships_gnn:
            tools.append(predict_relationships_gnn)
        if classify_nodes_gnn:
            tools.append(classify_nodes_gnn)
        if query_domain_gnn:
            tools.append(query_domain_gnn)
    
    # Add query router tools if available (Priority 3)
    if HAS_QUERY_ROUTER:
        if hybrid_query:
            tools.append(hybrid_query)
        if route_query:
            tools.append(route_query)
    
    # Add MCTS tools if available
    if HAS_MCTS_TOOLS:
        if plan_narrative_path_mcts:
            tools.append(plan_narrative_path_mcts)
        if what_if_analysis_mcts:
            tools.append(what_if_analysis_mcts)
        if reflective_mcts_debate:
            tools.append(reflective_mcts_debate)
    
    if custom_tools:
        tools.extend(custom_tools)
    
    # Create the deep agent
    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        **kwargs
    )
    
    return agent

