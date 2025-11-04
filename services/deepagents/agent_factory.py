"""Factory for creating DeepAgents with aModels-specific tools and prompts."""

import os
from typing import List, Optional
from langchain_core.tools import Tool
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent

from .tools import (
    query_knowledge_graph,
    run_agentflow_flow,
    run_orchestration_chain,
)


# Default system prompt for data pipeline and knowledge graph analysis
DEFAULT_SYSTEM_PROMPT = """You are an expert data pipeline analyst and knowledge graph specialist working with the aModels platform.

Your capabilities include:
- Querying and analyzing knowledge graphs (tables, schemas, lineage, data quality)
- Running AgentFlow/LangFlow workflows for data processing
- Executing orchestration chains for analysis and summarization
- Planning complex multi-step data pipeline tasks
- Breaking down complex problems into manageable sub-tasks

## Available Tools

### Knowledge Graph Analysis
- `query_knowledge_graph`: Execute Cypher queries against Neo4j to find tables, columns, relationships, SQL queries, Control-M jobs, and data quality metrics.

### AgentFlow Execution
- `run_agentflow_flow`: Run pre-configured LangFlow flows for data pipelines and processing workflows.

### Orchestration Chains
- `run_orchestration_chain`: Execute orchestration chains for:
  - Question answering with context
  - Summarization
  - Knowledge graph analysis
  - Data quality analysis
  - Pipeline analysis
  - SQL analysis

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
2. **Use Planning for Complex Tasks**: Break down multi-step analyses into todos.
3. **Leverage Sub-Agents**: Spawn specialized sub-agents for focused tasks.
4. **Combine Tools**: Use orchestration chains to analyze knowledge graph query results.
5. **Document Findings**: Use file system tools to save intermediate results and final reports.

## Example Workflow

For analyzing a data pipeline:
1. Query knowledge graph for Control-M jobs and SQL queries
2. Analyze dependencies and data flow
3. Use orchestration chain to assess data quality
4. Generate recommendations
5. Create AgentFlow flow for automated processing
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
        run_orchestration_chain,
    ]
    
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

