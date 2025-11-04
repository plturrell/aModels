"""Tool for querying the Neo4j knowledge graph."""

import os
from typing import Optional
import httpx
from langchain_core.tools import tool


EXTRACT_SERVICE_URL = os.getenv("EXTRACT_SERVICE_URL", "http://extract-service:19080")
_client = httpx.Client(timeout=30.0)


@tool
def query_knowledge_graph(
    query: str,
    project_id: Optional[str] = None,
    system_id: Optional[str] = None,
) -> str:
    """Query the Neo4j knowledge graph using Cypher queries.
    
    This tool allows you to query the knowledge graph to find:
    - Tables, columns, and their relationships
    - SQL queries and their lineage
    - Control-M jobs and dependencies
    - Data quality metrics
    - Pipeline structures
    
    Args:
        query: A Cypher query to execute against the knowledge graph.
               Example: "MATCH (n:Table) RETURN n LIMIT 10"
        project_id: Optional project ID to filter results
        system_id: Optional system ID to filter results
    
    Returns:
        JSON string with query results containing columns and data arrays
    
    Example queries:
    - "MATCH (n:Table) RETURN n.name, n.type LIMIT 10"
    - "MATCH (s:Table)-[r]->(t:Table) RETURN s.name, type(r), t.name LIMIT 20"
    - "MATCH (n:SQL) RETURN n.query LIMIT 5"
    """
    try:
        endpoint = f"{EXTRACT_SERVICE_URL}/knowledge-graph/query"
        
        payload = {"query": query}
        if project_id:
            payload.setdefault("params", {})["project_id"] = project_id
        if system_id:
            payload.setdefault("params", {})["system_id"] = system_id
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Format results for readability
        if "columns" in result and "data" in result:
            columns = result["columns"]
            data = result["data"]
            
            if not data:
                return "Query returned no results."
            
            # Format as readable text
            lines = [f"Columns: {', '.join(columns)}", ""]
            for i, row in enumerate(data[:10], 1):  # Limit to 10 rows
                lines.append(f"Row {i}:")
                for col in columns:
                    value = row.get(col, "N/A")
                    lines.append(f"  {col}: {value}")
                lines.append("")
            
            if len(data) > 10:
                lines.append(f"... and {len(data) - 10} more rows")
            
            return "\n".join(lines)
        
        return str(result)
    
    except httpx.HTTPStatusError as e:
        return f"Error querying knowledge graph: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error querying knowledge graph: {str(e)}"

