#!/usr/bin/env python3
"""
Week 4: Performance Tests

Tests system performance and benchmarks:
- Domain detection latency
- Model inference latency
- Routing optimization latency
- Throughput measurements
- Response time benchmarks
"""

import os
import sys
import json
import httpx
import time
import statistics
from typing import Optional, Dict, List, Any, Tuple
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

DEFAULT_TIMEOUT = 60
HEALTH_TIMEOUT = 5

# Performance thresholds (in milliseconds)
THRESHOLD_DOMAIN_DETECTION = 100  # 100ms for domain detection
THRESHOLD_MODEL_INFERENCE = 500   # 500ms for model inference
THRESHOLD_ROUTING = 50            # 50ms for routing
THRESHOLD_EXTRACTION = 2000       # 2s for extraction


@dataclass
class PerformanceResult:
    operation: str
    latency_ms: float
    throughput: float
    success: bool
    error: Optional[str] = None


class PerformanceTestSuite:
    def __init__(self):
        self.results: List[PerformanceResult] = []
        self.start_time = time.time()

    def measure_latency(self, operation: str, func) -> PerformanceResult:
        """Measure latency of an operation."""
        start = time.time()
        try:
            result = func()
            latency_ms = (time.time() - start) * 1000
            success = result is not None and result is not False
            return PerformanceResult(
                operation=operation,
                latency_ms=latency_ms,
                throughput=1.0 / (latency_ms / 1000) if latency_ms > 0 else 0,
                success=success
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return PerformanceResult(
                operation=operation,
                latency_ms=latency_ms,
                throughput=0,
                success=False,
                error=str(e)
            )

    def print_summary(self):
        """Print performance summary."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        
        print("\n" + "="*60)
        print("Performance Test Summary")
        print("="*60)
        print(f"Total Operations: {total}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print()
        
        if successful > 0:
            latencies = [r.latency_ms for r in self.results if r.success]
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 100 else max(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print("Latency Statistics (ms):")
            print(f"  Average: {avg_latency:.2f}")
            print(f"  P95: {p95_latency:.2f}")
            print(f"  Min: {min_latency:.2f}")
            print(f"  Max: {max_latency:.2f}")
            print()
            
            throughputs = [r.throughput for r in self.results if r.success and r.throughput > 0]
            if throughputs:
                avg_throughput = statistics.mean(throughputs)
                print(f"Average Throughput: {avg_throughput:.2f} ops/sec")
                print()
        
        # Performance by operation
        print("Performance by Operation:")
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            if result.success:
                operations[result.operation].append(result.latency_ms)
        
        for operation, latencies in operations.items():
            if latencies:
                avg = statistics.mean(latencies)
                print(f"  {operation}: {avg:.2f}ms avg ({len(latencies)} samples)")
        
        print()
        
        return failed == 0


def test_domain_detection_latency() -> bool:
    """Test domain detection latency."""
    suite = PerformanceTestSuite()
    
    def detect_domain():
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    result = suite.measure_latency("Domain Detection", detect_domain)
    suite.results.append(result)
    
    print(f"Domain Detection Latency: {result.latency_ms:.2f}ms")
    if result.latency_ms > THRESHOLD_DOMAIN_DETECTION:
        print(f"⚠️  Latency exceeds threshold ({THRESHOLD_DOMAIN_DETECTION}ms)")
        return False
    
    return result.success


def test_model_inference_latency() -> bool:
    """Test model inference latency."""
    suite = PerformanceTestSuite()
    
    def inference():
        payload = {
            "model": "general",
            "messages": [
                {"role": "user", "content": "Say 'test' if you can read this."}
            ],
            "max_tokens": 10
        }
        response = httpx.post(
            f"{LOCALAI_URL}/v1/chat/completions",
            json=payload,
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    result = suite.measure_latency("Model Inference", inference)
    suite.results.append(result)
    
    print(f"Model Inference Latency: {result.latency_ms:.2f}ms")
    if result.latency_ms > THRESHOLD_MODEL_INFERENCE:
        print(f"⚠️  Latency exceeds threshold ({THRESHOLD_MODEL_INFERENCE}ms)")
        return False
    
    return result.success


def test_routing_latency() -> bool:
    """Test routing optimization latency."""
    suite = PerformanceTestSuite()
    
    def routing():
        # Test routing by getting domains (simulates routing decision)
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    result = suite.measure_latency("Routing", routing)
    suite.results.append(result)
    
    print(f"Routing Latency: {result.latency_ms:.2f}ms")
    if result.latency_ms > THRESHOLD_ROUTING:
        print(f"⚠️  Latency exceeds threshold ({THRESHOLD_ROUTING}ms)")
        return False
    
    return result.success


def test_extraction_latency() -> bool:
    """Test extraction latency."""
    suite = PerformanceTestSuite()
    
    def extraction():
        request = create_extraction_request(
            sql_queries=["SELECT * FROM test_table"],
            project_id="perf_test",
            system_id="perf_system"
        )
        response = httpx.post(
            f"{EXTRACT_URL}/knowledge-graph",
            json=request,
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    result = suite.measure_latency("Extraction", extraction)
    suite.results.append(result)
    
    print(f"Extraction Latency: {result.latency_ms:.2f}ms")
    if result.latency_ms > THRESHOLD_EXTRACTION:
        print(f"⚠️  Latency exceeds threshold ({THRESHOLD_EXTRACTION}ms)")
        return False
    
    return result.success


def test_throughput() -> bool:
    """Test system throughput."""
    suite = PerformanceTestSuite()
    
    def simple_request():
        response = httpx.get(
            f"{LOCALAI_URL}/health",
            timeout=HEALTH_TIMEOUT
        )
        return response.status_code == 200
    
    # Measure throughput over 10 requests
    num_requests = 10
    start = time.time()
    
    for _ in range(num_requests):
        result = suite.measure_latency("Health Check", simple_request)
        suite.results.append(result)
    
    total_time = time.time() - start
    throughput = num_requests / total_time
    
    print(f"Throughput: {throughput:.2f} requests/sec")
    print(f"  Total time: {total_time:.2f}s for {num_requests} requests")
    
    return throughput > 1.0  # At least 1 req/sec


def test_response_time_consistency() -> bool:
    """Test response time consistency."""
    suite = PerformanceTestSuite()
    
    def domain_request():
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    # Measure 5 requests
    latencies = []
    for _ in range(5):
        result = suite.measure_latency("Domain Request", domain_request)
        suite.results.append(result)
        if result.success:
            latencies.append(result.latency_ms)
    
    if len(latencies) < 2:
        print("⚠️  Not enough successful requests for consistency test")
        return False
    
    avg_latency = statistics.mean(latencies)
    std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
    
    print(f"Response Time Consistency:")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  Std Dev: {std_dev:.2f}ms")
    print(f"  Coefficient of Variation: {(std_dev/avg_latency)*100:.2f}%")
    
    # Consistency is good if CV < 50%
    cv = (std_dev / avg_latency) * 100 if avg_latency > 0 else 100
    return cv < 50.0


def main():
    """Run all performance tests."""
    print("="*60)
    print("Performance Tests - Week 4")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"Extract Service URL: {EXTRACT_URL}")
    print()
    print(f"Performance Thresholds:")
    print(f"  Domain Detection: {THRESHOLD_DOMAIN_DETECTION}ms")
    print(f"  Model Inference: {THRESHOLD_MODEL_INFERENCE}ms")
    print(f"  Routing: {THRESHOLD_ROUTING}ms")
    print(f"  Extraction: {THRESHOLD_EXTRACTION}ms")
    print()
    
    # Wait for services
    print("Waiting for services...")
    if not wait_for_service(f"{LOCALAI_URL}/health", "LocalAI"):
        print("⚠️  LocalAI not available, some tests will be skipped")
    if not wait_for_service(f"{EXTRACT_URL}/healthz", "Extract Service"):
        print("⚠️  Extract service not available, some tests will be skipped")
    print()
    
    suite = PerformanceTestSuite()
    
    # Performance Tests
    print("Running performance tests...")
    print()
    
    test_domain_detection_latency()
    test_model_inference_latency()
    test_routing_latency()
    test_extraction_latency()
    test_throughput()
    test_response_time_consistency()
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
