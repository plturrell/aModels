"""Metrics and monitoring for TOON translation service."""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TranslationMetrics:
    """Metrics for a single translation request."""
    request_id: str
    model: str
    source_lang: str
    target_lang: str
    text_length: int
    toon_used: bool
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tokens_generated: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    
    def complete(self, tokens: int, success: bool = True, error: Optional[str] = None):
        """Mark the translation as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.tokens_generated = tokens
        self.success = success
        self.error = error


class TOONMetricsCollector:
    """Collects and aggregates metrics for TOON translation."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, TranslationMetrics] = {}
        self.aggregates: Dict[str, Any] = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration_ms": 0.0,
            "total_tokens": 0,
            "toon_usage_count": 0,
            "direct_translation_count": 0
        })
        self.request_counter = 0
    
    def start_request(
        self,
        model: str,
        source_lang: str,
        target_lang: str,
        text_length: int,
        use_toon: bool = False
    ) -> str:
        """Start tracking a translation request.
        
        Returns:
            Request ID
        """
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{int(time.time())}"
        
        metric = TranslationMetrics(
            request_id=request_id,
            model=model,
            source_lang=source_lang,
            target_lang=target_lang,
            text_length=text_length,
            toon_used=use_toon,
            start_time=time.time()
        )
        
        self.metrics[request_id] = metric
        return request_id
    
    def complete_request(
        self,
        request_id: str,
        tokens: int,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Mark a request as complete."""
        if request_id not in self.metrics:
            logger.warning(f"Request ID {request_id} not found in metrics")
            return
        
        metric = self.metrics[request_id]
        metric.complete(tokens, success, error)
        
        # Update aggregates
        key = f"{metric.model}:{metric.source_lang}:{metric.target_lang}"
        agg = self.aggregates[key]
        agg["total_requests"] += 1
        if success:
            agg["successful_requests"] += 1
        else:
            agg["failed_requests"] += 1
        
        if metric.duration_ms:
            agg["total_duration_ms"] += metric.duration_ms
        agg["total_tokens"] += tokens
        
        if metric.toon_used:
            agg["toon_usage_count"] += 1
        else:
            agg["direct_translation_count"] += 1
        
        # Log metrics
        logger.info(
            f"Translation completed: {request_id} | "
            f"Model: {metric.model} | "
            f"Duration: {metric.duration_ms:.2f}ms | "
            f"Tokens: {tokens} | "
            f"TOON: {'Yes' if metric.toon_used else 'No'} | "
            f"Success: {success}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        stats = {
            "total_requests": sum(agg["total_requests"] for agg in self.aggregates.values()),
            "successful_requests": sum(agg["successful_requests"] for agg in self.aggregates.values()),
            "failed_requests": sum(agg["failed_requests"] for agg in self.aggregates.values()),
            "by_model": {}
        }
        
        for key, agg in self.aggregates.items():
            model, src, tgt = key.split(":")
            model_key = f"{model}:{src}→{tgt}"
            
            avg_duration = 0.0
            if agg["total_requests"] > 0:
                avg_duration = agg["total_duration_ms"] / agg["total_requests"]
            
            success_rate = 0.0
            if agg["total_requests"] > 0:
                success_rate = agg["successful_requests"] / agg["total_requests"]
            
            stats["by_model"][model_key] = {
                "total_requests": agg["total_requests"],
                "successful": agg["successful_requests"],
                "failed": agg["failed_requests"],
                "success_rate": success_rate,
                "avg_duration_ms": avg_duration,
                "total_tokens": agg["total_tokens"],
                "toon_usage": agg["toon_usage_count"],
                "direct_translation": agg["direct_translation_count"],
                "toon_usage_rate": agg["toon_usage_count"] / max(agg["total_requests"], 1)
            }
        
        return stats
    
    def get_recent_requests(self, limit: int = 10) -> list[Dict[str, Any]]:
        """Get recent request metrics."""
        recent = sorted(
            self.metrics.values(),
            key=lambda m: m.start_time,
            reverse=True
        )[:limit]
        
        return [
            {
                "request_id": m.request_id,
                "model": m.model,
                "source_lang": m.source_lang,
                "target_lang": m.target_lang,
                "text_length": m.text_length,
                "toon_used": m.toon_used,
                "duration_ms": m.duration_ms,
                "tokens": m.tokens_generated,
                "success": m.success,
                "error": m.error
            }
            for m in recent
        ]
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get advanced analytics and reporting.
        
        Returns:
            Dictionary with analytics including:
            - Performance trends
            - Quality metrics
            - Usage patterns
            - Error analysis
        """
        stats = self.get_stats()
        
        # Calculate performance trends
        recent_requests = self.get_recent_requests(limit=100)
        if recent_requests:
            recent_durations = [r["duration_ms"] for r in recent_requests if r.get("duration_ms")]
            avg_recent_duration = sum(recent_durations) / len(recent_durations) if recent_durations else 0
            
            # Compare with overall average
            overall_avg = 0
            if stats["by_model"]:
                model_avgs = [m["avg_duration_ms"] for m in stats["by_model"].values() if m["avg_duration_ms"] > 0]
                overall_avg = sum(model_avgs) / len(model_avgs) if model_avgs else 0
            
            performance_trend = "improving" if avg_recent_duration < overall_avg else "stable" if avg_recent_duration == overall_avg else "degrading"
        else:
            performance_trend = "insufficient_data"
            avg_recent_duration = 0
        
        # Error analysis
        error_requests = [r for r in recent_requests if not r.get("success", True)]
        error_rate = len(error_requests) / len(recent_requests) if recent_requests else 0
        common_errors = {}
        for req in error_requests:
            error = req.get("error", "unknown")
            common_errors[error] = common_errors.get(error, 0) + 1
        
        # Usage patterns
        toon_usage_rate = sum(1 for r in recent_requests if r.get("toon_used", False)) / len(recent_requests) if recent_requests else 0
        
        return {
            "performance": {
                "trend": performance_trend,
                "recent_avg_duration_ms": avg_recent_duration,
                "overall_avg_duration_ms": overall_avg
            },
            "quality": {
                "success_rate": 1 - error_rate,
                "error_rate": error_rate,
                "toon_usage_rate": toon_usage_rate
            },
            "errors": {
                "total_errors": len(error_requests),
                "common_errors": dict(sorted(common_errors.items(), key=lambda x: x[1], reverse=True)[:5])
            },
            "usage": {
                "total_requests": stats["total_requests"],
                "toon_usage_rate": toon_usage_rate,
                "by_model": {
                    k: {
                        "requests": v["total_requests"],
                        "toon_usage": v["toon_usage_rate"]
                    }
                    for k, v in stats["by_model"].items()
                }
            }
        }


# Global metrics collector instance
_metrics_collector = TOONMetricsCollector()


def get_metrics_collector() -> TOONMetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector

