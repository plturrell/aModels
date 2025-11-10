"""Observability and metrics for DeepAgents service."""

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

# In-memory metrics storage (can be replaced with Prometheus/OpenTelemetry)
_metrics_lock = Lock()
_metrics = {
    "requests": defaultdict(int),
    "request_latency": defaultdict(list),
    "errors": defaultdict(int),
    "tool_usage": defaultdict(int),
    "token_usage": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    },
    "response_quality": {
        "structured_outputs": 0,
        "failed_structured": 0,
        "validation_errors": 0,
    },
}


def record_request(endpoint: str, method: str = "POST", status_code: int = 200):
    """Record a request."""
    with _metrics_lock:
        key = f"{method}:{endpoint}:{status_code}"
        _metrics["requests"][key] += 1


def record_latency(endpoint: str, latency_seconds: float):
    """Record request latency."""
    with _metrics_lock:
        _metrics["request_latency"][endpoint].append(latency_seconds)
        # Keep only last 1000 measurements per endpoint
        if len(_metrics["request_latency"][endpoint]) > 1000:
            _metrics["request_latency"][endpoint] = _metrics["request_latency"][endpoint][-1000:]


def record_error(endpoint: str, error_type: str):
    """Record an error."""
    with _metrics_lock:
        key = f"{endpoint}:{error_type}"
        _metrics["errors"][key] += 1


def record_tool_usage(tool_name: str):
    """Record tool usage."""
    with _metrics_lock:
        _metrics["tool_usage"][tool_name] += 1


def record_token_usage(input_tokens: int = 0, output_tokens: int = 0):
    """Record token usage."""
    with _metrics_lock:
        _metrics["token_usage"]["input_tokens"] += input_tokens
        _metrics["token_usage"]["output_tokens"] += output_tokens
        _metrics["token_usage"]["total_tokens"] += (input_tokens + output_tokens)


def record_structured_output(success: bool, validation_errors: int = 0):
    """Record structured output metrics."""
    with _metrics_lock:
        if success:
            _metrics["response_quality"]["structured_outputs"] += 1
        else:
            _metrics["response_quality"]["failed_structured"] += 1
        
        if validation_errors > 0:
            _metrics["response_quality"]["validation_errors"] += validation_errors


def get_metrics() -> Dict[str, Any]:
    """Get all metrics."""
    with _metrics_lock:
        # Calculate statistics
        latency_stats = {}
        for endpoint, latencies in _metrics["request_latency"].items():
            if latencies:
                latency_stats[endpoint] = {
                    "count": len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "avg": sum(latencies) / len(latencies),
                    "p50": _percentile(latencies, 0.5),
                    "p95": _percentile(latencies, 0.95),
                    "p99": _percentile(latencies, 0.99),
                }
        
        return {
            "requests": dict(_metrics["requests"]),
            "request_latency": latency_stats,
            "errors": dict(_metrics["errors"]),
            "tool_usage": dict(_metrics["tool_usage"]),
            "token_usage": dict(_metrics["token_usage"]),
            "response_quality": dict(_metrics["response_quality"]),
            "timestamp": datetime.utcnow().isoformat(),
        }


def _percentile(data: list, p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p)
    return sorted_data[min(index, len(sorted_data) - 1)]


def reset_metrics():
    """Reset all metrics (for testing)."""
    global _metrics
    with _metrics_lock:
        _metrics = {
            "requests": defaultdict(int),
            "request_latency": defaultdict(list),
            "errors": defaultdict(int),
            "tool_usage": defaultdict(int),
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "response_quality": {
                "structured_outputs": 0,
                "failed_structured": 0,
                "validation_errors": 0,
            },
        }


class MetricsMiddleware:
    """Middleware to track request metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        
        # Wrap send to capture status code
        status_code = 200
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
            
            # Record metrics
            latency = time.time() - start_time
            record_request(path, method, status_code)
            record_latency(path, latency)
            
            if status_code >= 400:
                error_type = "client_error" if status_code < 500 else "server_error"
                record_error(path, error_type)
        
        except Exception as e:
            latency = time.time() - start_time
            record_request(path, method, 500)
            record_latency(path, latency)
            record_error(path, "exception")
            raise

