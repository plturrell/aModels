"""Tool for querying and managing the catalog service."""

import os
from typing import Optional, List, Dict, Any
import httpx
from langchain_core.tools import tool
import json


CATALOG_SERVICE_URL = os.getenv("CATALOG_SERVICE_URL", "http://catalog-service:8084")
_client = httpx.Client(timeout=60.0)


@tool
def query_data_elements(
    name: Optional[str] = None,
    definition: Optional[str] = None,
    concept_id: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Query the catalog for data elements by name, definition, or concept ID.
    
    This tool allows you to search the catalog registry for existing data elements
    to check for duplicates, find similar elements, or understand existing metadata.
    
    Args:
        name: Optional name to search for (partial match)
        definition: Optional definition text to search for
        concept_id: Optional data element concept ID to filter by
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        JSON string with matching data elements including identifier, name, definition,
        concept_id, representation_id, and metadata.
    
    Examples:
        - Find elements by name: name="Customer ID"
        - Find elements by concept: concept_id="DEC-001"
        - Search definitions: definition="customer identifier"
    """
    try:
        # Use semantic search endpoint for flexible querying
        endpoint = f"{CATALOG_SERVICE_URL}/catalog/semantic-search"
        
        # Build query from provided parameters
        query_parts = []
        if name:
            query_parts.append(name)
        if definition:
            query_parts.append(definition)
        
        query = " ".join(query_parts) if query_parts else "*"
        
        payload = {
            "query": query,
        }
        if concept_id:
            # Note: semantic search doesn't directly support concept_id filter
            # but we can include it in the query
            payload["query"] = f"{query} {concept_id}"
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Format results
        if "results" in result and result["results"]:
            elements = result["results"][:limit]
            formatted = []
            for elem in elements:
                formatted.append({
                    "identifier": elem.get("identifier", "N/A"),
                    "name": elem.get("name", "N/A"),
                    "definition": elem.get("definition", "N/A"),
                    "data_element_concept_id": elem.get("data_element_concept_id", "N/A"),
                    "representation_id": elem.get("representation_id", "N/A"),
                    "metadata": elem.get("metadata", {}),
                })
            return json.dumps({"count": len(formatted), "elements": formatted}, indent=2)
        
        return json.dumps({"count": 0, "elements": []})
    
    except httpx.HTTPStatusError as e:
        return f"Error querying catalog: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error querying catalog: {str(e)}"


@tool
def check_duplicates(
    candidate_elements: List[Dict[str, Any]],
    existing_elements: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Check for duplicate data elements in the catalog.
    
    This tool analyzes candidate data elements and compares them against existing
    elements in the catalog to identify potential duplicates. Returns structured
    suggestions for each candidate.
    
    Args:
        candidate_elements: List of candidate elements to check, each with:
            - name: Element name
            - definition: Element definition
            - data_element_concept_id: Concept ID
            - representation_id: Representation ID
            - metadata: Optional metadata dict
        existing_elements: Optional list of existing elements to compare against.
                          If not provided, will query catalog for similar elements.
    
    Returns:
        JSON string with deduplication suggestions:
        {
            "suggestions": [
                {
                    "index": 0,
                    "action": "register" | "skip" | "merge",
                    "reason": "Explanation",
                    "similar_to": "identifier-of-similar-element",
                    "confidence": 0.0-1.0
                }
            ]
        }
    """
    try:
        # If existing elements not provided, query catalog for similar ones
        if existing_elements is None or len(existing_elements) == 0:
            # Query catalog for elements with similar names/concepts
            all_existing = []
            for candidate in candidate_elements:
                # Search by name
                name_results = _query_catalog_simple(candidate.get("name", ""))
                all_existing.extend(name_results)
                # Search by concept if available
                if candidate.get("data_element_concept_id"):
                    concept_results = _query_catalog_simple(
                        candidate.get("data_element_concept_id", "")
                    )
                    all_existing.extend(concept_results)
            existing_elements = all_existing
        
        # Build structured request
        request_data = {
            "candidate_elements": candidate_elements,
            "existing_elements": existing_elements,
        }
        
        # Call catalog's deduplication endpoint (if it exists) or use DeepAgents
        # For now, we'll return a structured format that can be processed
        # The actual deduplication logic will be handled by the agent using this tool
        
        suggestions = []
        for i, candidate in enumerate(candidate_elements):
            # Simple similarity check (can be enhanced)
            similar = _find_similar(candidate, existing_elements)
            if similar:
                suggestions.append({
                    "index": i,
                    "action": "skip",
                    "reason": f"Similar to existing element: {similar.get('name', 'unknown')}",
                    "similar_to": similar.get("identifier", ""),
                    "confidence": 0.7,  # Placeholder
                })
            else:
                suggestions.append({
                    "index": i,
                    "action": "register",
                    "reason": "No similar elements found",
                    "confidence": 0.9,
                })
        
        return json.dumps({"suggestions": suggestions}, indent=2)
    
    except Exception as e:
        return f"Error checking duplicates: {str(e)}"


@tool
def validate_definition(
    name: str,
    definition: str,
    concept_id: Optional[str] = None,
) -> str:
    """Validate a data element definition against ISO 11179 standards.
    
    This tool validates data element definitions for completeness, clarity,
    and compliance with ISO 11179 metadata standards.
    
    Args:
        name: Data element name
        definition: Data element definition text
        concept_id: Optional data element concept ID
    
    Returns:
        JSON string with validation results:
        {
            "score": 0.0-1.0,
            "improvements": ["suggestion1", "suggestion2"],
            "is_valid": true/false,
            "reason": "Explanation"
        }
    """
    try:
        # Basic validation checks
        score = 1.0
        improvements = []
        
        # Check definition length
        if len(definition) < 10:
            score -= 0.3
            improvements.append("Definition is too short (minimum 10 characters recommended)")
        
        if len(definition) > 500:
            score -= 0.1
            improvements.append("Definition is very long (consider breaking into multiple elements)")
        
        # Check for key components
        if not any(word in definition.lower() for word in ["data", "element", "represents", "indicates", "specifies"]):
            score -= 0.2
            improvements.append("Definition should clearly state what the element represents")
        
        # Check name clarity
        if len(name) < 3:
            score -= 0.2
            improvements.append("Name is too short")
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        result = {
            "score": score,
            "improvements": improvements,
            "is_valid": score >= 0.7,
            "reason": "Validation completed" if score >= 0.7 else "Definition needs improvement",
        }
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        return f"Error validating definition: {str(e)}"


@tool
def suggest_improvements(
    element: Dict[str, Any],
) -> str:
    """Suggest improvements to data element metadata.
    
    This tool analyzes a data element and suggests improvements to its
    name, definition, metadata, or other attributes.
    
    Args:
        element: Data element dict with name, definition, concept_id, etc.
    
    Returns:
        JSON string with improvement suggestions:
        {
            "improvements": [
                {
                    "field": "name" | "definition" | "metadata",
                    "suggestion": "Suggested improvement",
                    "priority": "high" | "medium" | "low"
                }
            ]
        }
    """
    try:
        improvements = []
        
        name = element.get("name", "")
        definition = element.get("definition", "")
        
        # Name improvements
        if not name:
            improvements.append({
                "field": "name",
                "suggestion": "Name is required",
                "priority": "high",
            })
        elif len(name) < 3:
            improvements.append({
                "field": "name",
                "suggestion": "Name should be at least 3 characters",
                "priority": "medium",
            })
        
        # Definition improvements
        if not definition:
            improvements.append({
                "field": "definition",
                "suggestion": "Definition is required",
                "priority": "high",
            })
        elif len(definition) < 20:
            improvements.append({
                "field": "definition",
                "suggestion": "Definition should be more detailed (minimum 20 characters)",
                "priority": "medium",
            })
        
        # Metadata suggestions
        metadata = element.get("metadata", {})
        if not metadata:
            improvements.append({
                "field": "metadata",
                "suggestion": "Consider adding metadata (source, steward, domain, etc.)",
                "priority": "low",
            })
        
        return json.dumps({"improvements": improvements}, indent=2)
    
    except Exception as e:
        return f"Error suggesting improvements: {str(e)}"


@tool
def find_similar_elements(
    name: str,
    definition: str,
    limit: int = 5,
) -> str:
    """Find similar existing elements in the catalog for context.
    
    This tool searches the catalog for elements similar to the provided
    name and definition, useful for finding related metadata or avoiding duplicates.
    
    Args:
        name: Element name to search for
        definition: Element definition to search for
        limit: Maximum number of similar elements to return (default: 5)
    
    Returns:
        JSON string with similar elements:
        {
            "count": 3,
            "elements": [
                {
                    "identifier": "...",
                    "name": "...",
                    "definition": "...",
                    "similarity_score": 0.85
                }
            ]
        }
    """
    try:
        # Use semantic search to find similar elements
        endpoint = f"{CATALOG_SERVICE_URL}/catalog/semantic-search"
        
        # Combine name and definition for search
        query = f"{name} {definition}"
        
        payload = {"query": query}
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Format results with similarity scores (simplified)
        if "results" in result and result["results"]:
            elements = result["results"][:limit]
            formatted = []
            for elem in elements:
                # Calculate simple similarity (can be enhanced)
                similarity = _calculate_similarity(name, definition, elem)
                formatted.append({
                    "identifier": elem.get("identifier", "N/A"),
                    "name": elem.get("name", "N/A"),
                    "definition": elem.get("definition", "N/A"),
                    "data_element_concept_id": elem.get("data_element_concept_id", "N/A"),
                    "similarity_score": similarity,
                })
            
            # Sort by similarity
            formatted.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return json.dumps({"count": len(formatted), "elements": formatted}, indent=2)
        
        return json.dumps({"count": 0, "elements": []})
    
    except httpx.HTTPStatusError as e:
        return f"Error finding similar elements: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error finding similar elements: {str(e)}"


# Helper functions

def _query_catalog_simple(query: str) -> List[Dict[str, Any]]:
    """Helper to query catalog with a simple text query."""
    try:
        endpoint = f"{CATALOG_SERVICE_URL}/catalog/semantic-search"
        payload = {"query": query}
        response = _client.post(endpoint, json=payload, timeout=10.0)
        response.raise_for_status()
        result = response.json()
        return result.get("results", [])
    except Exception:
        return []


def _find_similar(candidate: Dict[str, Any], existing: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find similar element in existing list."""
    candidate_name = candidate.get("name", "").lower()
    candidate_def = candidate.get("definition", "").lower()
    
    best_match = None
    best_score = 0.0
    
    for elem in existing:
        elem_name = elem.get("name", "").lower()
        elem_def = elem.get("definition", "").lower()
        
        # Simple similarity: name match or definition overlap
        name_sim = 1.0 if candidate_name == elem_name else 0.0
        def_sim = _text_similarity(candidate_def, elem_def)
        
        score = (name_sim * 0.7) + (def_sim * 0.3)
        
        if score > best_score and score > 0.5:  # Threshold
            best_score = score
            best_match = elem
    
    return best_match


def _calculate_similarity(name: str, definition: str, element: Dict[str, Any]) -> float:
    """Calculate similarity score between query and element."""
    elem_name = element.get("name", "").lower()
    elem_def = element.get("definition", "").lower()
    
    name_lower = name.lower()
    def_lower = definition.lower()
    
    # Name similarity
    if name_lower == elem_name:
        name_score = 1.0
    elif name_lower in elem_name or elem_name in name_lower:
        name_score = 0.7
    else:
        name_score = _text_similarity(name_lower, elem_name)
    
    # Definition similarity
    def_score = _text_similarity(def_lower, elem_def)
    
    # Combined score
    return (name_score * 0.6) + (def_score * 0.4)


def _text_similarity(text1: str, text2: str) -> float:
    """Simple text similarity using word overlap."""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

