"""Metrics collection for integration calls."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Simple in-memory metrics store (in production, use Prometheus or similar)
_metrics_store: dict[str, list[float]] = defaultdict(list)
_request_counts: dict[str, int] = defaultdict(int)
_error_counts: dict[str, int] = defaultdict(int)
_circuit_breaker_states: dict[str, str] = {}
_retry_counts: dict[str, int] = defaultdict(int)


def collect_integration_metric(
    service: str,
    endpoint: str,
    status_code: int,
    latency: float,
    correlation_id: Optional[str] = None,
) -> None:
    """Collect metrics for an integration call.
    
    Args:
        service: Service name (e.g., "extract", "catalog")
        endpoint: Endpoint path
        status_code: HTTP status code
        latency: Request latency in seconds
        correlation_id: Optional correlation ID for tracing
    """
    key = f"{service}:{endpoint}"
    
    # Store latency
    _metrics_store[key].append(latency)
    # Keep only last 1000 measurements
    if len(_metrics_store[key]) > 1000:
        _metrics_store[key] = _metrics_store[key][-1000:]
    
    # Increment request count
    _request_counts[key] += 1
    
    # Track errors
    if status_code >= 400:
        _error_counts[key] += 1
        error_key = f"{key}:{status_code}"
        _error_counts[error_key] += 1
    
    # Log with correlation ID if available
    if correlation_id:
        logger.debug(
            "[%s] Integration metric: %s %s -> %d (latency: %.3fs)",
            correlation_id,
            service,
            endpoint,
            status_code,
            latency,
        )


def record_circuit_breaker_state(service: str, state: str) -> None:
    """Record circuit breaker state change.
    
    Args:
        service: Service name
        state: Circuit breaker state ("closed", "open", "half_open")
    """
    _circuit_breaker_states[service] = state
    logger.warning("Circuit breaker state changed: %s -> %s", service, state)


def record_retry(service: str, endpoint: str) -> None:
    """Record a retry attempt.
    
    Args:
        service: Service name
        endpoint: Endpoint path
    """
    key = f"{service}:{endpoint}"
    _retry_counts[key] += 1


def get_metrics_summary() -> dict:
    """Get summary of all metrics.
    
    Returns:
        Dictionary with metrics summary
    """
    summary = {}
    
    for key, latencies in _metrics_store.items():
        if not latencies:
            continue
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        summary[key] = {
            "request_count": _request_counts.get(key, 0),
            "error_count": _error_counts.get(key, 0),
            "retry_count": _retry_counts.get(key, 0),
            "latency": {
                "p50": sorted_latencies[n // 2] if n > 0 else 0.0,
                "p95": sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[0] if n > 0 else 0.0,
                "p99": sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[0] if n > 0 else 0.0,
                "mean": sum(latencies) / n if n > 0 else 0.0,
            },
        }
    
    summary["circuit_breaker_states"] = dict(_circuit_breaker_states)
    
    return summary


def get_error_rate(service: str, endpoint: str) -> float:
    """Get error rate for a service endpoint.
    
    Args:
        service: Service name
        endpoint: Endpoint path
    
    Returns:
        Error rate (0.0 to 1.0)
    """
    key = f"{service}:{endpoint}"
    request_count = _request_counts.get(key, 0)
    if request_count == 0:
        return 0.0
    error_count = _error_counts.get(key, 0)
    return error_count / request_count

