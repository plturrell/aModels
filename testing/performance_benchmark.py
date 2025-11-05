#!/usr/bin/env python3
"""
Week 4: Performance Benchmark Suite

Comprehensive performance benchmarking:
- Baseline performance metrics
- Performance regression detection
- Performance comparison across components
- Resource utilization tracking
"""

import os
import sys
import json
import httpx
import time
import statistics
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path

# Add test helpers to path
sys.path.insert(0, os.path.dirname(__file__))
from test_helpers import (
    check_service_health, wait_for_service, print_test_summary
)

# Test configuration
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
EXTRACT_URL = os.getenv("EXTRACT_SERVICE_URL", "http://localhost:19080")

DEFAULT_TIMEOUT = 60
HEALTH_TIMEOUT = 5

# Benchmark parameters
BENCHMARK_ITERATIONS = 10
WARMUP_ITERATIONS = 2

# Performance baselines (in milliseconds)
BASELINES = {
    "domain_detection": 100,
    "model_inference": 500,
    "routing": 50,
    "extraction": 2000,
    "embedding": 200,
}


class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.latencies: List[float] = []
        self.successful = 0
        self.failed = 0
        self.errors: List[str] = []
    
    def add_result(self, latency_ms: float, success: bool, error: Optional[str] = None):
        if success:
            self.latencies.append(latency_ms)
            self.successful += 1
        else:
            self.failed += 1
            if error:
                self.errors.append(error)
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.latencies:
            return {
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "throughput": 0
            }
        
        avg = statistics.mean(self.latencies)
        p95 = statistics.quantiles(self.latencies, n=100)[94] if len(self.latencies) >= 100 else max(self.latencies)
        p99 = statistics.quantiles(self.latencies, n=100)[98] if len(self.latencies) >= 100 else max(self.latencies)
        min_lat = min(self.latencies)
        max_lat = max(self.latencies)
        
        # Calculate throughput (requests per second)
        total_time = sum(self.latencies) / 1000
        throughput = len(self.latencies) / total_time if total_time > 0 else 0
        
        return {
            "avg_latency_ms": avg,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "min_latency_ms": min_lat,
            "max_latency_ms": max_lat,
            "throughput": throughput
        }
    
    def compare_to_baseline(self, baseline: float) -> Tuple[bool, float]:
        """Compare performance to baseline. Returns (is_better, improvement_percent)."""
        if not self.latencies:
            return False, 0.0
        
        avg_latency = statistics.mean(self.latencies)
        improvement = ((baseline - avg_latency) / baseline) * 100
        is_better = avg_latency <= baseline
        
        return is_better, improvement


class PerformanceBenchmark:
    def __init__(self):
        self.results: Dict[str, BenchmarkResult] = {}
        self.start_time = time.time()
    
    def run_benchmark(self, name: str, operation, iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS):
        """Run a benchmark."""
        print(f"Benchmarking: {name}")
        print(f"  Iterations: {iterations} (warmup: {warmup})")
        
        result = BenchmarkResult(name)
        
        # Warmup
        for _ in range(warmup):
            try:
                operation()
            except Exception:
                pass
        
        # Actual benchmark
        for i in range(iterations):
            start = time.time()
            try:
                success = operation()
                latency_ms = (time.time() - start) * 1000
                result.add_result(latency_ms, success)
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                result.add_result(latency_ms, False, str(e))
        
        self.results[name] = result
        
        # Print results
        stats = result.get_stats()
        baseline = BASELINES.get(name, 0)
        
        print(f"  ✅ Successful: {result.successful}/{iterations}")
        print(f"  ❌ Failed: {result.failed}/{iterations}")
        print(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"  P95 Latency: {stats['p95_latency_ms']:.2f}ms")
        print(f"  Throughput: {stats['throughput']:.2f} req/sec")
        
        if baseline > 0:
            is_better, improvement = result.compare_to_baseline(baseline)
            status = "✅" if is_better else "⚠️"
            print(f"  {status} Baseline: {baseline}ms (Improvement: {improvement:+.2f}%)")
        
        print()
        
        return result
    
    def print_summary(self):
        """Print benchmark summary."""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("Performance Benchmark Summary")
        print("="*60)
        print(f"Total Benchmarks: {len(self.results)}")
        print(f"Total Duration: {total_time:.2f}s")
        print()
        
        for name, result in self.results.items():
            stats = result.get_stats()
            baseline = BASELINES.get(name, 0)
            
            print(f"{name}:")
            print(f"  Success Rate: {(result.successful/(result.successful+result.failed))*100:.2f}%")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
            print(f"  P95 Latency: {stats['p95_latency_ms']:.2f}ms")
            print(f"  Throughput: {stats['throughput']:.2f} req/sec")
            
            if baseline > 0:
                is_better, improvement = result.compare_to_baseline(baseline)
                status = "✅" if is_better else "⚠️"
                print(f"  {status} vs Baseline: {improvement:+.2f}%")
            
            print()
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save benchmark results to file."""
        output_dir = Path(__file__).parent / "benchmarks"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_{timestamp}.json"
        
        results_data = {}
        for name, result in self.results.items():
            results_data[name] = {
                "stats": result.get_stats(),
                "successful": result.successful,
                "failed": result.failed,
                "errors": list(set(result.errors))
            }
        
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✅ Benchmark results saved to: {output_file}")


def benchmark_domain_detection(benchmark: PerformanceBenchmark):
    """Benchmark domain detection."""
    def detect_domain():
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    return benchmark.run_benchmark("domain_detection", detect_domain)


def benchmark_model_inference(benchmark: PerformanceBenchmark):
    """Benchmark model inference."""
    def inference():
        payload = {
            "model": "general",
            "messages": [
                {"role": "user", "content": "Say 'test'."}
            ],
            "max_tokens": 10
        }
        response = httpx.post(
            f"{LOCALAI_URL}/v1/chat/completions",
            json=payload,
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    return benchmark.run_benchmark("model_inference", inference)


def benchmark_routing(benchmark: PerformanceBenchmark):
    """Benchmark routing."""
    def routing():
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    return benchmark.run_benchmark("routing", routing)


def benchmark_extraction(benchmark: PerformanceBenchmark):
    """Benchmark extraction."""
    from test_helpers import create_extraction_request
    
    def extraction():
        request = create_extraction_request(
            sql_queries=["SELECT * FROM test_table"],
            project_id="benchmark_test",
            system_id="benchmark_system"
        )
        response = httpx.post(
            f"{EXTRACT_URL}/knowledge-graph",
            json=request,
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    return benchmark.run_benchmark("extraction", extraction)


def benchmark_embedding(benchmark: PerformanceBenchmark):
    """Benchmark embedding generation."""
    def embedding():
        payload = {
            "model": "0x3579-VectorProcessingAgent",
            "input": ["test embedding"]
        }
        response = httpx.post(
            f"{LOCALAI_URL}/v1/embeddings",
            json=payload,
            timeout=DEFAULT_TIMEOUT
        )
        return response.status_code == 200
    
    return benchmark.run_benchmark("embedding", embedding)


def main():
    """Run all performance benchmarks."""
    print("="*60)
    print("Performance Benchmark Suite - Week 4")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"Extract Service URL: {EXTRACT_URL}")
    print()
    print(f"Benchmark Parameters:")
    print(f"  Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Warmup: {WARMUP_ITERATIONS}")
    print()
    print(f"Performance Baselines:")
    for name, baseline in BASELINES.items():
        print(f"  {name}: {baseline}ms")
    print()
    
    # Wait for services
    print("Waiting for services...")
    if not wait_for_service(f"{LOCALAI_URL}/health", "LocalAI"):
        print("⚠️  LocalAI not available, some benchmarks will be skipped")
    if not wait_for_service(f"{EXTRACT_URL}/healthz", "Extract Service"):
        print("⚠️  Extract service not available, some benchmarks will be skipped")
    print()
    
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    print("Running performance benchmarks...")
    print()
    
    benchmark_domain_detection(benchmark)
    benchmark_model_inference(benchmark)
    benchmark_routing(benchmark)
    benchmark_extraction(benchmark)
    benchmark_embedding(benchmark)
    
    # Print summary
    benchmark.print_summary()
    
    sys.exit(0)


if __name__ == "__main__":
    main()

