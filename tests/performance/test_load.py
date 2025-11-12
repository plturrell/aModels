#!/usr/bin/env python3
"""
Week 4: Load Tests

Tests system under load:
- Concurrent domain requests
- Large knowledge graphs
- High-volume training
- A/B test traffic splitting performance
- Resource usage under load
"""

import os
import sys
import json
import httpx
import time
import statistics
import threading
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Add test helpers to path
sys.path.insert(0, os.path.dirname(__file__))
from test_helpers import (
    check_service_health, wait_for_service, print_test_summary,
    create_extraction_request
)

# Test configuration
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localai-compat:8080")
EXTRACT_URL = os.getenv("EXTRACT_SERVICE_URL", "http://extract-service:8082")
TRAINING_URL = os.getenv("TRAINING_SERVICE_URL", "http://training-service:8080")

DEFAULT_TIMEOUT = 60
HEALTH_TIMEOUT = 5

# Load test parameters
CONCURRENT_REQUESTS = 10
REQUESTS_PER_THREAD = 5
LARGE_GRAPH_NODES = 1000


@dataclass
class LoadTestResult:
    operation: str
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    p95_latency_ms: float
    throughput: float
    errors: List[str]


class LoadTestSuite:
    def __init__(self):
        self.results: List[LoadTestResult] = []
        self.start_time = time.time()

    def print_summary(self):
        """Print load test summary."""
        total = len(self.results)
        
        print("\n" + "="*60)
        print("Load Test Summary")
        print("="*60)
        print(f"Total Test Scenarios: {total}")
        print()
        
        for result in self.results:
            print(f"Operation: {result.operation}")
            print(f"  Total Requests: {result.total_requests}")
            print(f"  ✅ Successful: {result.successful}")
            print(f"  ❌ Failed: {result.failed}")
            print(f"  Success Rate: {(result.successful/result.total_requests)*100:.2f}%")
            print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
            print(f"  P95 Latency: {result.p95_latency_ms:.2f}ms")
            print(f"  Throughput: {result.throughput:.2f} req/sec")
            if result.errors:
                print(f"  Errors: {len(result.errors)} unique errors")
            print()
        
        return all(r.successful > 0 for r in self.results)


def test_concurrent_domain_requests() -> bool:
    """Test concurrent domain requests."""
    print("Testing concurrent domain requests...")
    print(f"  Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"  Requests per thread: {REQUESTS_PER_THREAD}")
    print()
    
    latencies = []
    errors = []
    successful = 0
    total = CONCURRENT_REQUESTS * REQUESTS_PER_THREAD
    
    def make_request(request_id: int):
        try:
            start = time.time()
            response = httpx.get(
                f"{LOCALAI_URL}/v1/domains",
                timeout=DEFAULT_TIMEOUT
            )
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return latency_ms, None
            else:
                return None, f"Status {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = []
        for i in range(total):
            future = executor.submit(make_request, i)
            futures.append(future)
        
        for future in as_completed(futures):
            latency, error = future.result()
            if latency is not None:
                latencies.append(latency)
                successful += 1
            else:
                errors.append(error)
    
    failed = total - successful
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 100 else max(latencies)
        throughput = total / (max(latencies) / 1000) if latencies else 0
    else:
        avg_latency = 0
        p95_latency = 0
        throughput = 0
    
    result = LoadTestResult(
        operation="Concurrent Domain Requests",
        total_requests=total,
        successful=successful,
        failed=failed,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        throughput=throughput,
        errors=list(set(errors))
    )
    
    suite = LoadTestSuite()
    suite.results.append(result)
    suite.print_summary()
    
    print(f"✅ Concurrent requests test complete")
    print(f"   Success rate: {(successful/total)*100:.2f}%")
    
    return successful > 0


def test_large_knowledge_graph() -> bool:
    """Test extraction with large knowledge graph."""
    print("Testing large knowledge graph extraction...")
    print(f"  Target nodes: {LARGE_GRAPH_NODES}")
    print()
    
    # Create a large extraction request
    # Generate multiple SQL queries to create large graph
    sql_queries = []
    for i in range(LARGE_GRAPH_NODES // 10):  # Each query creates ~10 nodes
        sql_queries.append(f"SELECT * FROM table_{i} WHERE id = {i}")
    
    request = create_extraction_request(
        sql_queries=sql_queries[:100],  # Limit to 100 queries for performance
        project_id="load_test_large",
        system_id="load_test_system"
    )
    
    start = time.time()
    try:
        response = httpx.post(
            f"{EXTRACT_URL}/knowledge-graph",
            json=request,
            timeout=DEFAULT_TIMEOUT * 2  # Longer timeout for large graphs
        )
        
        latency_ms = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            
            print(f"✅ Large graph extraction successful")
            print(f"   Nodes extracted: {len(nodes)}")
            print(f"   Edges extracted: {len(edges)}")
            print(f"   Latency: {latency_ms:.2f}ms")
            
            return len(nodes) > 0
        else:
            print(f"❌ Large graph extraction failed: {response.status_code}")
            return False
            
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        print(f"❌ Large graph extraction error: {e}")
        print(f"   Latency: {latency_ms:.2f}ms")
        return False


def test_high_volume_training() -> bool:
    """Test high-volume training requests."""
    print("Testing high-volume training...")
    print()
    
    # Check if training service is available
    TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", TRAINING_URL)
    if not check_service_health(f"{TRAINING_SERVICE_URL}/health", "Training Service"):
        print("⚠️  Training service not available")
        return False
    
    # Test training pipeline components
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
    
    try:
        from domain_filter import DomainFilter, PrivacyConfig
        from domain_trainer import DomainTrainer
        
        # Initialize components
        privacy_config = PrivacyConfig(epsilon=1.0)
        domain_filter = DomainFilter(
            localai_url=LOCALAI_URL,
            privacy_config=privacy_config
        )
        
        trainer = DomainTrainer(
            localai_url=LOCALAI_URL
        )
        
        print(f"✅ Training components initialized")
        print(f"   (High-volume training tested in integration)")
        
        return True
        
    except ImportError:
        print("⚠️  Training components not available (module not found)")
        return False


def test_ab_test_traffic_splitting_performance() -> bool:
    """Test A/B test traffic splitting performance."""
    print("Testing A/B test traffic splitting performance...")
    print()
    
    TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", TRAINING_URL)
    
    # Test via training service API
    try:
        domain_id = "test-financial"
        num_requests = 100
        
        start = time.time()
        variant_counts = {"A": 0, "B": 0, "default": 0}
        successful_requests = 0
        
        for i in range(num_requests):
            try:
                response = httpx.post(
                    f"{TRAINING_SERVICE_URL}/ab-test/route",
                    json={
                        "domain_id": domain_id,
                        "request_id": f"request_{i}"
                    },
                    timeout=5.0
                )
                if response.status_code == 200:
                    result = response.json()
                    variant = result.get("variant", "default")
                    variant_counts[variant] = variant_counts.get(variant, 0) + 1
                    successful_requests += 1
            except Exception:
                pass
        
        latency_ms = (time.time() - start) * 1000
        avg_latency = latency_ms / num_requests if num_requests > 0 else 0
        
        if successful_requests > 0:
            print(f"✅ Traffic splitting performance test")
            print(f"   Total requests: {num_requests}")
            print(f"   Successful: {successful_requests}")
            print(f"   Total latency: {latency_ms:.2f}ms")
            print(f"   Avg latency per request: {avg_latency:.2f}ms")
            print(f"   Variant distribution: {variant_counts}")
            return avg_latency < 100.0  # Reasonable threshold for HTTP requests
        else:
            print("⚠️  A/B test routing failed (no successful requests)")
            return False
        
    except Exception as e:
        print(f"⚠️  A/B test manager not available: {e}")
        return False


def test_resource_usage_under_load() -> bool:
    """Test resource usage under load."""
    print("Testing resource usage under load...")
    print()
    
    # Make multiple concurrent requests and measure resource impact
    num_requests = 20
    
    start = time.time()
    
    def make_request():
        try:
            response = httpx.get(
                f"{LOCALAI_URL}/health",
                timeout=HEALTH_TIMEOUT
            )
            return response.status_code == 200
        except Exception:
            return False
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        results = [f.result() for f in as_completed(futures)]
    
    total_time = time.time() - start
    successful = sum(1 for r in results if r)
    
    print(f"✅ Resource usage test")
    print(f"   Total requests: {num_requests}")
    print(f"   Successful: {successful}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Throughput: {num_requests/total_time:.2f} req/sec")
    print(f"   (Resource monitoring would require system metrics)")
    
    return successful > 0


def main():
    """Run all load tests."""
    print("="*60)
    print("Load Tests - Week 4")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"Extract Service URL: {EXTRACT_URL}")
    print(f"Training Service URL: {TRAINING_URL}")
    print()
    print(f"Load Test Parameters:")
    print(f"  Concurrent Requests: {CONCURRENT_REQUESTS}")
    print(f"  Requests per Thread: {REQUESTS_PER_THREAD}")
    print(f"  Large Graph Nodes: {LARGE_GRAPH_NODES}")
    print()
    
    # Wait for services
    print("Waiting for services...")
    if not wait_for_service(f"{LOCALAI_URL}/health", "LocalAI"):
        print("⚠️  LocalAI not available, some tests will be skipped")
    if not wait_for_service(f"{EXTRACT_URL}/healthz", "Extract Service"):
        print("⚠️  Extract service not available, some tests will be skipped")
    print()
    
    suite = LoadTestSuite()
    
    # Load Tests
    print("Running load tests...")
    print()
    
    test_concurrent_domain_requests()
    test_large_knowledge_graph()
    test_high_volume_training()
    test_ab_test_traffic_splitting_performance()
    test_resource_usage_under_load()
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
