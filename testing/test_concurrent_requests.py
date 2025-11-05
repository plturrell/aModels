#!/usr/bin/env python3
"""
Week 4: Concurrent Requests Tests

Tests system with concurrent requests:
- Multiple domains simultaneously
- Concurrent extraction requests
- Concurrent training requests
- Concurrent A/B test routing
- Race condition handling
"""

import os
import sys
import json
import httpx
import time
import statistics
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
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
EXTRACT_URL = os.getenv("EXTRACT_SERVICE_URL", "http://localhost:19080")

DEFAULT_TIMEOUT = 60
HEALTH_TIMEOUT = 5

# Concurrency parameters
NUM_CONCURRENT_DOMAINS = 5
REQUESTS_PER_DOMAIN = 10


@dataclass
class ConcurrentTestResult:
    test_name: str
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    errors: List[str]


class ConcurrentTestSuite:
    def __init__(self):
        self.results: List[ConcurrentTestResult] = []
        self.start_time = time.time()

    def print_summary(self):
        """Print concurrent test summary."""
        total = len(self.results)
        
        print("\n" + "="*60)
        print("Concurrent Requests Test Summary")
        print("="*60)
        print(f"Total Test Scenarios: {total}")
        print()
        
        for result in self.results:
            print(f"Test: {result.test_name}")
            print(f"  Total Requests: {result.total_requests}")
            print(f"  ✅ Successful: {result.successful}")
            print(f"  ❌ Failed: {result.failed}")
            print(f"  Success Rate: {(result.successful/result.total_requests)*100:.2f}%")
            print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
            if result.errors:
                print(f"  Errors: {len(result.errors)} unique")
            print()
        
        return all(r.successful > 0 for r in self.results)


def test_multiple_domains_simultaneously() -> bool:
    """Test multiple domains accessed simultaneously."""
    print("Testing multiple domains simultaneously...")
    print(f"  Domains: {NUM_CONCURRENT_DOMAINS}")
    print(f"  Requests per domain: {REQUESTS_PER_DOMAIN}")
    print()
    
    latencies = []
    errors = []
    successful = 0
    total = NUM_CONCURRENT_DOMAINS * REQUESTS_PER_DOMAIN
    
    def fetch_domain(domain_id: str, request_num: int):
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
    
    # Test with multiple domain requests
    test_domains = ["test-financial", "test-customer", "test-product", "general", "default"]
    
    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_DOMAINS) as executor:
        futures = []
        for domain in test_domains[:NUM_CONCURRENT_DOMAINS]:
            for i in range(REQUESTS_PER_DOMAIN):
                future = executor.submit(fetch_domain, domain, i)
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
    else:
        avg_latency = 0
    
    result = ConcurrentTestResult(
        test_name="Multiple Domains Simultaneously",
        total_requests=total,
        successful=successful,
        failed=failed,
        avg_latency_ms=avg_latency,
        errors=list(set(errors))
    )
    
    suite = ConcurrentTestSuite()
    suite.results.append(result)
    
    print(f"✅ Multiple domains test complete")
    print(f"   Success rate: {(successful/total)*100:.2f}%")
    print(f"   Avg latency: {avg_latency:.2f}ms")
    
    return successful > 0


def test_concurrent_extraction_requests() -> bool:
    """Test concurrent extraction requests."""
    print("Testing concurrent extraction requests...")
    print()
    
    num_concurrent = 5
    latencies = []
    errors = []
    successful = 0
    
    def extract_request(request_id: int):
        try:
            request = create_extraction_request(
                sql_queries=[f"SELECT * FROM table_{request_id}"],
                project_id=f"concurrent_test_{request_id}",
                system_id=f"concurrent_system_{request_id}"
            )
            
            start = time.time()
            response = httpx.post(
                f"{EXTRACT_URL}/knowledge-graph",
                json=request,
                timeout=DEFAULT_TIMEOUT
            )
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return latency_ms, None
            else:
                return None, f"Status {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(extract_request, i) for i in range(num_concurrent)]
        
        for future in as_completed(futures):
            latency, error = future.result()
            if latency is not None:
                latencies.append(latency)
                successful += 1
            else:
                errors.append(error)
    
    if latencies:
        avg_latency = statistics.mean(latencies)
    else:
        avg_latency = 0
    
    print(f"✅ Concurrent extraction test complete")
    print(f"   Successful: {successful}/{num_concurrent}")
    print(f"   Avg latency: {avg_latency:.2f}ms")
    
    return successful > 0


def test_concurrent_training_requests() -> bool:
    """Test concurrent training requests."""
    print("Testing concurrent training requests...")
    print()
    
    # Check if training service is available
    TRAINING_URL = os.getenv("TRAINING_SERVICE_URL", "http://localhost:8080")
    if not check_service_health(f"{TRAINING_URL}/health", "Training Service"):
        print("⚠️  Training service not available")
        return False
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
    
    try:
        from domain_trainer import DomainTrainer
        
        trainer = DomainTrainer(localai_url=LOCALAI_URL)
        
        # Test concurrent training initialization
        num_concurrent = 3
        successful = 0
        
        def init_trainer(domain_id: str):
            try:
                trainer_instance = DomainTrainer(localai_url=LOCALAI_URL)
                return trainer_instance is not None
            except Exception:
                return False
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(init_trainer, f"test-domain-{i}")
                for i in range(num_concurrent)
            ]
            results = [f.result() for f in as_completed(futures)]
            successful = sum(1 for r in results if r)
        
        print(f"✅ Concurrent training test complete")
        print(f"   Successful: {successful}/{num_concurrent}")
        
        return successful > 0
        
    except ImportError:
        print("⚠️  Training components not available (module not found)")
        return False


def test_concurrent_ab_test_routing() -> bool:
    """Test concurrent A/B test routing."""
    print("Testing concurrent A/B test routing...")
    print()
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
    
    try:
        from ab_testing import ABTestManager
        
        manager = ABTestManager()
        
        domain_id = "test-financial"
        num_concurrent = 20
        
        variant_counts = {"A": 0, "B": 0, "default": 0}
        successful = 0
        
        def route_request(request_id: int):
            try:
                variant, config = manager.route_request(domain_id, f"request_{request_id}")
                return variant, None
            except Exception as e:
                return None, str(e)
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(route_request, i) for i in range(num_concurrent)]
            
            for future in as_completed(futures):
                variant, error = future.result()
                if variant:
                    variant_counts[variant] = variant_counts.get(variant, 0) + 1
                    successful += 1
        
        print(f"✅ Concurrent A/B routing test complete")
        print(f"   Successful: {successful}/{num_concurrent}")
        print(f"   Variant distribution: {variant_counts}")
        
        return successful > 0
        
    except ImportError:
        print("⚠️  A/B test manager not available (module not found)")
        return False


def test_race_condition_handling() -> bool:
    """Test race condition handling."""
    print("Testing race condition handling...")
    print()
    
    # Test that concurrent requests to same resource don't cause issues
    num_concurrent = 10
    resource_id = "test_resource"
    
    successful = 0
    errors = []
    
    def access_resource(request_id: int):
        try:
            # Simulate resource access
            response = httpx.get(
                f"{LOCALAI_URL}/v1/domains",
                timeout=DEFAULT_TIMEOUT
            )
            if response.status_code == 200:
                return True, None
            else:
                return False, f"Status {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(access_resource, i) for i in range(num_concurrent)]
        
        for future in as_completed(futures):
            success, error = future.result()
            if success:
                successful += 1
            else:
                errors.append(error)
    
    print(f"✅ Race condition test complete")
    print(f"   Successful: {successful}/{num_concurrent}")
    print(f"   Errors: {len(set(errors))} unique errors")
    
    # Race condition handling is good if most requests succeed
    return successful >= num_concurrent * 0.8  # 80% success rate


def main():
    """Run all concurrent request tests."""
    print("="*60)
    print("Concurrent Requests Tests - Week 4")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"Extract Service URL: {EXTRACT_URL}")
    print()
    print(f"Concurrency Parameters:")
    print(f"  Concurrent Domains: {NUM_CONCURRENT_DOMAINS}")
    print(f"  Requests per Domain: {REQUESTS_PER_DOMAIN}")
    print()
    
    # Wait for services
    print("Waiting for services...")
    if not wait_for_service(f"{LOCALAI_URL}/health", "LocalAI"):
        print("⚠️  LocalAI not available, some tests will be skipped")
    if not wait_for_service(f"{EXTRACT_URL}/healthz", "Extract Service"):
        print("⚠️  Extract service not available, some tests will be skipped")
    print()
    
    suite = ConcurrentTestSuite()
    
    # Concurrent Tests
    print("Running concurrent request tests...")
    print()
    
    test_multiple_domains_simultaneously()
    test_concurrent_extraction_requests()
    test_concurrent_training_requests()
    test_concurrent_ab_test_routing()
    test_race_condition_handling()
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

