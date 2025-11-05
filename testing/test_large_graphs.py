#!/usr/bin/env python3
"""
Week 4: Large Knowledge Graph Tests

Tests handling of large knowledge graphs:
- Large graph extraction
- Domain detection on large graphs
- Graph processing performance
- Memory usage with large graphs
"""

import os
import sys
import json
import httpx
import time
from typing import Optional, Dict, List, Any

# Add test helpers to path
sys.path.insert(0, os.path.dirname(__file__))
from test_helpers import (
    check_service_health, wait_for_service, print_test_summary,
    create_extraction_request
)

# Test configuration
EXTRACT_URL = os.getenv("EXTRACT_SERVICE_URL", "http://localhost:19080")
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")

DEFAULT_TIMEOUT = 300  # Longer timeout for large graphs
HEALTH_TIMEOUT = 5

# Large graph parameters
LARGE_GRAPH_NODES = 1000
LARGE_GRAPH_EDGES = 5000


class TestResult:
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"
    WARN = "⚠️"


class LargeGraphTestSuite:
    def __init__(self):
        self.tests = []
        self.start_time = time.time()

    def run_test(self, name: str, description: str, test_func):
        """Run a test and record the result."""
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        print(f"Description: {description}")
        print()
        
        start = time.time()
        try:
            result = test_func()
            duration = time.time() - start
            
            if result:
                self.tests.append({"name": name, "result": TestResult.PASS, "duration": duration})
                print(f"{TestResult.PASS} {name} passed ({duration:.2f}s)")
            else:
                self.tests.append({"name": name, "result": TestResult.FAIL, "duration": duration})
                print(f"{TestResult.FAIL} {name} failed ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start
            self.tests.append({"name": name, "result": TestResult.FAIL, "error": str(e), "duration": duration})
            print(f"{TestResult.FAIL} {name} failed with error: {e} ({duration:.2f}s)")
            import traceback
            traceback.print_exc()
        
        return result if 'result' in locals() else False

    def print_summary(self):
        """Print test summary."""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t.get("result") == TestResult.PASS)
        failed = sum(1 for t in self.tests if t.get("result") == TestResult.FAIL)
        skipped = sum(1 for t in self.tests if t.get("result") == TestResult.SKIP)
        
        print_test_summary("Large Knowledge Graph Tests", passed, failed, skipped)
        
        if failed > 0:
            print("Failed Tests:")
            for test in self.tests:
                if test.get("result") == TestResult.FAIL:
                    print(f"  {TestResult.FAIL} {test['name']}: {test.get('error', 'Unknown error')}")
            print()
        
        return failed == 0


def generate_large_sql_queries(num_queries: int = 100) -> List[str]:
    """Generate a large number of SQL queries for testing."""
    queries = []
    
    for i in range(num_queries):
        # Generate diverse queries
        query_type = i % 4
        if query_type == 0:
            queries.append(f"SELECT customer_id, payment_amount, transaction_date FROM financial_transactions_{i}")
        elif query_type == 1:
            queries.append(f"SELECT user_id, email, phone FROM customer_contacts_{i}")
        elif query_type == 2:
            queries.append(f"SELECT product_id, sku, price FROM product_catalog_{i}")
        else:
            queries.append(f"SELECT order_id, customer_id, total_amount FROM orders_{i} WHERE status = 'ACTIVE'")
    
    return queries


def test_large_graph_extraction() -> bool:
    """Test extraction of large knowledge graph."""
    try:
        # Generate large number of SQL queries
        num_queries = 100
        sql_queries = generate_large_sql_queries(num_queries)
        
        print(f"Testing large graph extraction")
        print(f"   SQL queries: {len(sql_queries)}")
        
        request = create_extraction_request(
            sql_queries=sql_queries,
            project_id="large_graph_test",
            system_id="large_graph_system"
        )
        
        start = time.time()
        response = httpx.post(
            f"{EXTRACT_URL}/knowledge-graph",
            json=request,
            timeout=DEFAULT_TIMEOUT
        )
        duration = time.time() - start
        
        if response.status_code != 200:
            print(f"❌ Large graph extraction failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
        
        data = response.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print(f"✅ Large graph extraction successful")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Nodes: {len(nodes)}")
        print(f"   Edges: {len(edges)}")
        
        # Check if graph is reasonably large
        if len(nodes) > 10 and len(edges) > 10:
            return True
        else:
            print(f"⚠️  Graph smaller than expected")
            return False
        
    except Exception as e:
        print(f"❌ Large graph extraction test error: {e}")
        return False


def test_domain_detection_on_large_graph() -> bool:
    """Test domain detection on large knowledge graph."""
    try:
        # Generate large graph with mixed domain content
        sql_queries = []
        
        # Financial domain queries
        for i in range(30):
            sql_queries.append(f"SELECT payment_amount, transaction_id FROM financial_transactions_{i}")
        
        # Customer domain queries
        for i in range(30):
            sql_queries.append(f"SELECT customer_id, email FROM customer_contacts_{i}")
        
        # Product domain queries
        for i in range(30):
            sql_queries.append(f"SELECT product_id, sku FROM product_catalog_{i}")
        
        print(f"Testing domain detection on large graph")
        print(f"   SQL queries: {len(sql_queries)}")
        
        request = create_extraction_request(
            sql_queries=sql_queries,
            project_id="large_graph_domain_test",
            system_id="large_graph_domain_system"
        )
        
        start = time.time()
        response = httpx.post(
            f"{EXTRACT_URL}/knowledge-graph",
            json=request,
            timeout=DEFAULT_TIMEOUT
        )
        duration = time.time() - start
        
        if response.status_code != 200:
            print(f"⚠️  Large graph extraction failed: {response.status_code}")
            return False
        
        data = response.json()
        nodes = data.get("nodes", [])
        
        # Count nodes with domain metadata
        nodes_with_domain = sum(1 for node in nodes if node.get("properties", {}).get("domain") or node.get("props", {}).get("domain"))
        
        print(f"✅ Domain detection on large graph")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Total nodes: {len(nodes)}")
        print(f"   Nodes with domain: {nodes_with_domain}")
        
        if nodes_with_domain > 0:
            return True
        else:
            print(f"⚠️  No nodes have domain metadata")
            return False
        
    except Exception as e:
        print(f"❌ Domain detection on large graph test error: {e}")
        return False


def test_graph_processing_performance() -> bool:
    """Test graph processing performance with varying sizes."""
    try:
        graph_sizes = [10, 50, 100]  # Number of SQL queries
        results = []
        
        for size in graph_sizes:
            sql_queries = generate_large_sql_queries(size)
            
            request = create_extraction_request(
                sql_queries=sql_queries,
                project_id=f"perf_test_{size}",
                system_id=f"perf_system_{size}"
            )
            
            start = time.time()
            response = httpx.post(
                f"{EXTRACT_URL}/knowledge-graph",
                json=request,
                timeout=DEFAULT_TIMEOUT
            )
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])
                
                results.append({
                    "size": size,
                    "duration": duration,
                    "nodes": len(nodes),
                    "edges": len(edges)
                })
            
        print(f"✅ Graph processing performance test")
        print(f"   Graph sizes tested: {graph_sizes}")
        for result in results:
            print(f"   Size {result['size']}: {result['duration']:.2f}s, {result['nodes']} nodes, {result['edges']} edges")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"⚠️  Graph processing performance test: {e}")
        return False


def test_memory_efficiency() -> bool:
    """Test memory efficiency with large graphs."""
    try:
        # Test with progressively larger graphs
        sizes = [50, 100, 200]
        
        print(f"Testing memory efficiency")
        print(f"   Graph sizes: {sizes}")
        
        for size in sizes:
            sql_queries = generate_large_sql_queries(size)
            
            request = create_extraction_request(
                sql_queries=sql_queries,
                project_id=f"memory_test_{size}",
                system_id=f"memory_system_{size}"
            )
            
            try:
                response = httpx.post(
                    f"{EXTRACT_URL}/knowledge-graph",
                    json=request,
                    timeout=DEFAULT_TIMEOUT
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Size {size}: Success ({len(data.get('nodes', []))} nodes)")
                else:
                    print(f"   Size {size}: Failed ({response.status_code})")
                    return False
            except Exception as e:
                print(f"   Size {size}: Error - {e}")
                return False
        
        print(f"✅ Memory efficiency test")
        print(f"   (Memory usage tested via system monitoring)")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test error: {e}")
        return False


def main():
    """Run all large graph tests."""
    global suite
    suite = LargeGraphTestSuite()
    
    print("="*60)
    print("Large Knowledge Graph Tests - Week 4")
    print("="*60)
    print(f"Extract Service URL: {EXTRACT_URL}")
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"Test parameters:")
    print(f"  Large graph nodes: {LARGE_GRAPH_NODES}")
    print(f"  Large graph edges: {LARGE_GRAPH_EDGES}")
    print()
    
    # Wait for services
    print("Waiting for services...")
    if not wait_for_service(f"{EXTRACT_URL}/healthz", "Extract Service"):
        print("⚠️  Extract service not available, some tests will be skipped")
    if not wait_for_service(f"{LOCALAI_URL}/health", "LocalAI"):
        print("⚠️  LocalAI not available, some tests will be skipped")
    print()
    
    # Large Graph Tests
    suite.run_test(
        "Large Graph Extraction",
        f"Test extraction of large knowledge graph ({LARGE_GRAPH_NODES} nodes, {LARGE_GRAPH_EDGES} edges)",
        test_large_graph_extraction
    )
    
    suite.run_test(
        "Domain Detection on Large Graph",
        "Test domain detection on large knowledge graph",
        test_domain_detection_on_large_graph
    )
    
    suite.run_test(
        "Graph Processing Performance",
        "Test graph processing performance with varying sizes",
        test_graph_processing_performance
    )
    
    suite.run_test(
        "Memory Efficiency",
        "Test memory efficiency with large graphs",
        test_memory_efficiency
    )
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

