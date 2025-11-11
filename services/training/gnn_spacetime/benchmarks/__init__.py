"""Performance benchmarks for narrative spacetime GNN system."""

from .performance_benchmarks import (
    benchmark_explanation_generation,
    benchmark_prediction_accuracy,
    benchmark_anomaly_detection,
    benchmark_runtime_performance
)

__all__ = [
    "benchmark_explanation_generation",
    "benchmark_prediction_accuracy",
    "benchmark_anomaly_detection",
    "benchmark_runtime_performance",
]

