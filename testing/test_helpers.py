#!/usr/bin/env python3
"""
Week 2: Test Helpers

Shared utilities for integration tests:
- Service health checks
- Test data loading
- Assertion helpers
- Domain verification
"""

import os
import json
import httpx
from typing import Dict, List, Optional, Any
from pathlib import Path

# Test configuration
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
EXTRACT_URL = os.getenv("EXTRACT_SERVICE_URL", "http://localhost:19080")
TRAINING_URL = os.getenv("TRAINING_SERVICE_URL", "http://localhost:8080")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:pass@localhost:5432/amodels")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")

DEFAULT_TIMEOUT = 30
HEALTH_TIMEOUT = 5

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def check_service_health(url: str, name: str, timeout: int = HEALTH_TIMEOUT) -> bool:
    """Check if a service is healthy."""
    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def load_test_data(filename: str) -> Dict[str, Any]:
    """Load test data from JSON file."""
    filepath = TEST_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Test data file not found: {filepath}")
    
    with open(filepath, "r") as f:
        return json.load(f)


def verify_domain_in_node(node: Dict[str, Any], expected_domain: Optional[str] = None) -> bool:
    """Verify that a node has domain metadata."""
    props = node.get("properties", node.get("props", {}))
    
    has_domain = "domain" in props or "domain_id" in props
    has_agent_id = "agent_id" in props
    
    if expected_domain:
        domain = props.get("domain") or props.get("domain_id")
        return domain == expected_domain and has_agent_id
    
    return has_domain and has_agent_id


def verify_domain_in_edge(edge: Dict[str, Any], expected_domain: Optional[str] = None) -> bool:
    """Verify that an edge has domain metadata."""
    props = edge.get("properties", edge.get("props", {}))
    
    has_domain = "domain" in props or "domain_id" in props
    has_agent_id = "agent_id" in props
    
    if expected_domain:
        domain = props.get("domain") or props.get("domain_id")
        return domain == expected_domain and has_agent_id
    
    return has_domain and has_agent_id


def count_nodes_by_domain(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count nodes by domain."""
    domain_counts = {}
    
    for node in nodes:
        props = node.get("properties", node.get("props", {}))
        domain = props.get("domain") or props.get("domain_id") or "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    return domain_counts


def count_edges_by_domain(edges: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count edges by domain."""
    domain_counts = {}
    
    for edge in edges:
        props = edge.get("properties", edge.get("props", {}))
        domain = props.get("domain") or props.get("domain_id") or "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    return domain_counts


def get_domain_from_localai(domain_id: str) -> Optional[Dict[str, Any]]:
    """Get domain configuration from LocalAI."""
    try:
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            domains = data.get("data", [])
            
            for domain in domains:
                if domain.get("id") == domain_id:
                    return domain.get("config", domain)
        
        return None
    except Exception:
        return None


def create_extraction_request(
    json_tables: Optional[List[str]] = None,
    hive_ddls: Optional[List[str]] = None,
    sql_queries: Optional[List[str]] = None,
    project_id: str = "test_project",
    system_id: str = "test_system"
) -> Dict[str, Any]:
    """Create a sample extraction request."""
    return {
        "json_tables": json_tables or [],
        "hive_ddls": hive_ddls or [],
        "sql_queries": sql_queries or [],
        "project_id": project_id,
        "system_id": system_id,
        "information_system_id": system_id
    }


def create_training_request(
    domain_id: str,
    training_data_path: str,
    base_model_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create a sample training request."""
    return {
        "domain_id": domain_id,
        "training_data_path": training_data_path,
        "base_model_path": base_model_path,
        "fine_tune": True
    }


def wait_for_service(url: str, name: str, max_attempts: int = 10, delay: int = 2) -> bool:
    """Wait for a service to become available."""
    for attempt in range(max_attempts):
        if check_service_health(url, name):
            return True
        if attempt < max_attempts - 1:
            import time
            time.sleep(delay)
    
    return False


def verify_extraction_response(response: Dict[str, Any]) -> bool:
    """Verify that an extraction response has the expected structure."""
    required_fields = ["nodes", "edges"]
    
    for field in required_fields:
        if field not in response:
            return False
    
    nodes = response.get("nodes", [])
    edges = response.get("edges", [])
    
    # Verify nodes and edges are lists
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False
    
    return True


def verify_training_response(response: Dict[str, Any]) -> bool:
    """Verify that a training response has the expected structure."""
    required_fields = ["domain_id", "training_run_id"]
    
    for field in required_fields:
        if field not in response:
            return False
    
    return True


def extract_domains_from_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> set:
    """Extract all unique domains from a knowledge graph."""
    domains = set()
    
    for node in nodes:
        props = node.get("properties", node.get("props", {}))
        domain = props.get("domain") or props.get("domain_id")
        if domain:
            domains.add(domain)
    
    for edge in edges:
        props = edge.get("properties", edge.get("props", {}))
        domain = props.get("domain") or props.get("domain_id")
        if domain:
            domains.add(domain)
    
    return domains


def assert_domain_detected(graph_response: Dict[str, Any], expected_domain: Optional[str] = None) -> bool:
    """Assert that at least one domain was detected in the graph."""
    nodes = graph_response.get("nodes", [])
    edges = graph_response.get("edges", [])
    
    detected_domains = extract_domains_from_graph(nodes, edges)
    
    if expected_domain:
        return expected_domain in detected_domains
    
    return len(detected_domains) > 0


def print_test_summary(test_name: str, passed: int, failed: int, skipped: int):
    """Print a test summary."""
    total = passed + failed + skipped
    print(f"\n{'='*60}")
    print(f"{test_name} Summary")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"{'='*60}\n")

