#!/usr/bin/env python3
"""
NumPy Baseline Benchmark Script
Compares Go maths package performance against NumPy baseline
"""

import numpy as np
import time
import sys
import os
from typing import List, Tuple, Dict
import json

class NumPyBenchmark:
    def __init__(self):
        self.results = {}
        
    def benchmark_matrix_multiplication(self, sizes: List[int]) -> Dict[str, float]:
        """Benchmark matrix multiplication for various sizes"""
        results = {}
        
        for size in sizes:
            # Create test matrices
            A = np.random.randn(size, size).astype(np.float64)
            B = np.random.randn(size, size).astype(np.float64)
            
            # Warm up
            _ = np.dot(A, B)
            
            # Benchmark
            times = []
            for _ in range(5):
                start = time.perf_counter()
                C = np.dot(A, B)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            results[f"matmul_{size}x{size}"] = avg_time * 1000  # Convert to milliseconds
            
        return results
    
    def benchmark_element_wise_operations(self, sizes: List[int]) -> Dict[str, float]:
        """Benchmark element-wise operations"""
        results = {}
        
        for size in sizes:
            A = np.random.randn(size).astype(np.float64)
            B = np.random.randn(size).astype(np.float64)
            
            operations = {
                'add': lambda a, b: a + b,
                'multiply': lambda a, b: a * b,
                'sqrt': lambda a, b: np.sqrt(a),
                'sin': lambda a, b: np.sin(a),
                'tanh': lambda a, b: np.tanh(a),
            }
            
            for op_name, op_func in operations.items():
                # Warm up
                _ = op_func(A, B)
                
                # Benchmark
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    _ = op_func(A, B)
                    end = time.perf_counter()
                    times.append(end - start)
                
                avg_time = np.mean(times)
                results[f"{op_name}_{size}"] = avg_time * 1000  # Convert to milliseconds
                
        return results
    
    def benchmark_vector_operations(self, sizes: List[int]) -> Dict[str, float]:
        """Benchmark vector operations (dot product, cosine similarity)"""
        results = {}
        
        for size in sizes:
            A = np.random.randn(size).astype(np.float64)
            B = np.random.randn(size).astype(np.float64)
            
            # Dot product
            times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = np.dot(A, B)
                end = time.perf_counter()
                times.append(end - start)
            results[f"dot_{size}"] = np.mean(times) * 1000
            
            # Cosine similarity
            times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
                end = time.perf_counter()
                times.append(end - start)
            results[f"cosine_{size}"] = np.mean(times) * 1000
            
        return results
    
    def benchmark_statistical_operations(self, sizes: List[int]) -> Dict[str, float]:
        """Benchmark statistical operations"""
        results = {}
        
        for size in sizes:
            data = np.random.randn(size).astype(np.float64)
            
            operations = {
                'sum': lambda x: np.sum(x),
                'mean': lambda x: np.mean(x),
                'std': lambda x: np.std(x),
                'min': lambda x: np.min(x),
                'max': lambda x: np.max(x),
                'argmax': lambda x: np.argmax(x),
            }
            
            for op_name, op_func in operations.items():
                # Warm up
                _ = op_func(data)
                
                # Benchmark
                times = []
                for _ in range(50):
                    start = time.perf_counter()
                    _ = op_func(data)
                    end = time.perf_counter()
                    times.append(end - start)
                
                avg_time = np.mean(times)
                results[f"{op_name}_{size}"] = avg_time * 1000
                
        return results
    
    def benchmark_fft_operations(self, sizes: List[int]) -> Dict[str, float]:
        """Benchmark FFT operations"""
        results = {}
        
        for size in sizes:
            data = np.random.randn(size).astype(np.float64)
            
            # 1D FFT
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = np.fft.fft(data)
                end = time.perf_counter()
                times.append(end - start)
            results[f"fft_1d_{size}"] = np.mean(times) * 1000
            
            # 2D FFT
            if size <= 1024:  # Limit 2D FFT size
                data_2d = np.random.randn(size, size).astype(np.float64)
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    _ = np.fft.fft2(data_2d)
                    end = time.perf_counter()
                    times.append(end - start)
                results[f"fft_2d_{size}x{size}"] = np.mean(times) * 1000
                
        return results
    
    def run_all_benchmarks(self) -> Dict[str, float]:
        """Run all benchmark suites"""
        print("Running NumPy baseline benchmarks...")
        
        # Matrix multiplication benchmarks
        print("  Matrix multiplication...")
        matrix_sizes = [64, 256, 512, 1024]
        self.results.update(self.benchmark_matrix_multiplication(matrix_sizes))
        
        # Element-wise operations
        print("  Element-wise operations...")
        element_sizes = [1000, 10000, 100000, 1000000]
        self.results.update(self.benchmark_element_wise_operations(element_sizes))
        
        # Vector operations
        print("  Vector operations...")
        vector_sizes = [100, 1000, 10000, 100000]
        self.results.update(self.benchmark_vector_operations(vector_sizes))
        
        # Statistical operations
        print("  Statistical operations...")
        stat_sizes = [1000, 10000, 100000]
        self.results.update(self.benchmark_statistical_operations(stat_sizes))
        
        # FFT operations
        print("  FFT operations...")
        fft_sizes = [1024, 4096, 16384]
        self.results.update(self.benchmark_fft_operations(fft_sizes))
        
        return self.results
    
    def save_results(self, filename: str = "numpy_baseline.json"):
        """Save benchmark results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of benchmark results"""
        print("\n" + "="*60)
        print("NumPy Baseline Benchmark Results")
        print("="*60)
        
        # Group results by category
        categories = {
            'Matrix Operations': [k for k in self.results.keys() if k.startswith('matmul_')],
            'Element-wise Operations': [k for k in self.results.keys() if any(k.startswith(op) for op in ['add_', 'multiply_', 'sqrt_', 'sin_', 'tanh_'])],
            'Vector Operations': [k for k in self.results.keys() if any(k.startswith(op) for op in ['dot_', 'cosine_'])],
            'Statistical Operations': [k for k in self.results.keys() if any(k.startswith(op) for op in ['sum_', 'mean_', 'std_', 'min_', 'max_', 'argmax_'])],
            'FFT Operations': [k for k in self.results.keys() if k.startswith('fft_')],
        }
        
        for category, operations in categories.items():
            if operations:
                print(f"\n{category}:")
                print("-" * len(category))
                for op in sorted(operations):
                    time_ms = self.results[op]
                    print(f"  {op:30} {time_ms:8.3f} ms")
        
        print(f"\nTotal operations benchmarked: {len(self.results)}")
        print("="*60)

def main():
    """Main benchmark execution"""
    print("NumPy Baseline Benchmark")
    print("=" * 40)
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Create benchmark instance
    benchmark = NumPyBenchmark()
    
    # Run all benchmarks
    results = benchmark.run_all_benchmarks()
    
    # Save results
    benchmark.save_results()
    
    # Print summary
    benchmark.print_summary()
    
    # Generate comparison report
    print("\nGenerating comparison report...")
    generate_comparison_report(results)

def generate_comparison_report(numpy_results: Dict[str, float]):
    """Generate a comparison report with Go maths package claims"""
    
    # Go maths package performance claims (from README)
    go_claims = {
        'matmul_1024x1024': 0.08,  # 6.25x faster than NumPy
        'add_1000000': 0.3,        # 6.7x faster than NumPy  
        'mean_100000': 0.2,        # 5x faster than NumPy
    }
    
    print("\n" + "="*60)
    print("Performance Comparison: Go Maths vs NumPy")
    print("="*60)
    
    for operation, go_time in go_claims.items():
        if operation in numpy_results:
            numpy_time = numpy_results[operation]
            speedup = numpy_time / go_time
            print(f"{operation:30} NumPy: {numpy_time:8.3f}ms  Go: {go_time:6.3f}ms  Speedup: {speedup:5.1f}x")
        else:
            print(f"{operation:30} No NumPy baseline available")
    
    print("="*60)
    
    # Save comparison to file
    comparison = {
        'numpy_baseline': numpy_results,
        'go_claims': go_claims,
        'timestamp': time.time(),
        'numpy_version': np.__version__,
    }
    
    with open('performance_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("Comparison saved to performance_comparison.json")

if __name__ == "__main__":
    main()
