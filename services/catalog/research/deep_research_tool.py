"""
Open Deep Research tool for catalog metadata research.
Integrates langchain-ai/open_deep_research for intelligent metadata discovery.
"""
import os
import json
import httpx
from typing import Dict, Any, List, Optional

# Open Deep Research API endpoint (can be local or hosted)
DEEP_RESEARCH_URL = os.getenv("DEEP_RESEARCH_URL", "http://localhost:8085")
CATALOG_SPARQL_URL = os.getenv("CATALOG_SPARQL_URL", "http://localhost:8084/catalog/sparql")

async def research_metadata(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Use Open Deep Research to research metadata questions.
    
    Args:
        query: Research question (e.g., "What data elements exist for customer data?")
        context: Optional context (e.g., domain, source system)
    
    Returns:
        Research report with findings
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Prepare research request
        payload = {
            "query": query,
            "context": context or {},
            "tools": ["sparql_query", "catalog_search"],  # Available tools
        }
        
        try:
            response = await client.post(
                f"{DEEP_RESEARCH_URL}/research",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Research failed: {e.response.status_code}",
                "status": "error",
                "report": None
            }

async def query_catalog_sparql(sparql_query: str) -> Dict[str, Any]:
    """
    Execute SPARQL query against catalog triplestore.
    Used by Open Deep Research as a tool.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                CATALOG_SPARQL_URL,
                json={"query": sparql_query},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"SPARQL query failed: {e.response.status_code}",
                "results": {"bindings": []}
            }

async def search_catalog_semantic(query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Semantic search in catalog.
    Used by Open Deep Research as a tool.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            payload = {
                "query": query,
                **(filters or {})
            }
            response = await client.post(
                CATALOG_SPARQL_URL.replace("/sparql", "/semantic-search"),
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Semantic search failed: {e.response.status_code}",
                "results": []
            }

async def generate_metadata_report(
    topic: str,
    include_lineage: bool = True,
    include_quality: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive metadata research report using Open Deep Research.
    
    This is the "complete data product" - a research report that answers:
    - What data exists for this topic?
    - Where does it come from?
    - What is its quality?
    - How can it be used?
    """
    research_query = f"""
    Research and document all metadata related to: {topic}
    
    Include:
    1. Data elements and their definitions
    2. Data lineage (sources, transformations)
    3. Quality metrics and SLOs
    4. Access controls and permissions
    5. Usage patterns and examples
    6. Related data products
    """
    
    # Use Open Deep Research to generate report
    report = await research_metadata(research_query, {
        "topic": topic,
        "include_lineage": include_lineage,
        "include_quality": include_quality
    })
    
    return report

