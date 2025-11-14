from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
from redis import asyncio as aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from retry_utils import retry_http_request
from circuit_breaker import CircuitBreakerManager, CircuitBreakerOpenError, CircuitBreakerConfig
from rate_limiter import MultiTierRateLimiter
from logging_config import setup_logging, get_logger, set_correlation_id, get_correlation_id
from proxy_utils import ServiceProxy, create_proxy_helper
from cache_manager import CacheManager, CacheStrategy
from models import (
    UnifiedSearchRequest, UnifiedSearchResponse, SearchResult, SearchMetadata,
    ProcessRequest, ProcessResponse, StatusResponse, HistoryRequest, HistoryResponse,
    BatchRequest, BatchResponse, DataElementRequest, DataProductRequest,
    SemanticSearchRequest, GraphQueryRequest, ResearchRequest,
    HealthResponse, ServiceHealth, CircuitBreakerStats, CacheStats,
    GenericResponse, ErrorDetail
)

# Setup structured logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JSON_LOGS = os.getenv("JSON_LOGS", "false").lower() == "true"
setup_logging(level=LOG_LEVEL, json_format=JSON_LOGS)

# Get structured logger
logger = get_logger(__name__)


GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))

HANA_URL = os.getenv("HANA_URL", "http://localhost:8083")
AGENTFLOW_URL = os.getenv("AGENTFLOW_URL", "http://localhost:9001")
EXTRACT_URL = os.getenv("EXTRACT_URL", "http://localhost:9002")
DATA_URL = os.getenv("DATA_URL", "http://localhost:9003")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
SEARCH_INFERENCE_URL = os.getenv("SEARCH_INFERENCE_URL", "http://localhost:8090")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8080")
BROWSER_URL = os.getenv("BROWSER_URL", "http://localhost:8070")
DEEPAGENTS_URL = os.getenv("DEEPAGENTS_URL", "http://localhost:9004")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8081")
SAP_BDC_URL = os.getenv("SAP_BDC_URL", "http://localhost:8083")
CATALOG_URL = os.getenv("CATALOG_URL", "http://localhost:8084")
DEEP_RESEARCH_URL = os.getenv("DEEP_RESEARCH_URL", "http://localhost:8085")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

if LOCALAI_URL == GRAPH_SERVICE_URL:
    logger.warning(
        "LOCALAI_URL matches GRAPH_SERVICE_URL - potential routing conflict",
        localai_url=LOCALAI_URL,
        graph_url=GRAPH_SERVICE_URL
    )

# Initialize global clients (will be set in lifespan)
client: httpx.AsyncClient | None = None
redis_client: aioredis.Redis | None = None
circuit_breaker_manager: CircuitBreakerManager | None = None
rate_limiter: MultiTierRateLimiter | None = None
cache_manager: CacheManager | None = None
service_proxy: ServiceProxy | None = None

# Prometheus metrics
request_counter = Counter('gateway_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('gateway_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
circuit_breaker_state = Gauge('gateway_circuit_breaker_state', 'Circuit breaker state', ['service'])
rate_limit_rejections = Counter('gateway_rate_limit_rejections_total', 'Rate limit rejections', ['endpoint'])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global client, redis_client, circuit_breaker_manager, rate_limiter, cache_manager, service_proxy
    
    logger.info("Starting aModels Gateway v0.2.0", port=GATEWAY_PORT)
    
    # Initialize httpx client with connection pooling and proper timeout
    client = httpx.AsyncClient(
        timeout=60.0,  # Increased from 30s
        limits=httpx.Limits(
            max_keepalive_connections=20,  # Keep 20 connections alive
            max_connections=100,  # Max 100 total connections
            keepalive_expiry=30.0  # Keep alive for 30s
        )
    )
    logger.info("HTTP client initialized", timeout=60.0, max_connections=100)
    
    # Initialize Redis
    try:
        redis_client = aioredis.from_url(REDIS_URL)
        await redis_client.ping()
        logger.info("Redis connected", url=REDIS_URL)
    except Exception as e:
        logger.warning("Redis connection failed, continuing without cache", error=str(e))
        redis_client = None
    
    # Initialize circuit breaker manager
    circuit_breaker_manager = CircuitBreakerManager()
    logger.info("Circuit breaker manager initialized")
    
    # Initialize rate limiter
    rate_limiter = MultiTierRateLimiter()
    await rate_limiter.start_cleanup_task()
    logger.info("Rate limiter initialized", 
                default_rate="100 req/min",
                strict_rate="20 req/min",
                lenient_rate="300 req/min")
    
    # Initialize cache manager
    cache_manager = CacheManager(redis_client)
    logger.info("Cache manager initialized", 
                enabled=redis_client is not None,
                strategies=["SHORT: 60s", "MEDIUM: 5m", "LONG: 1h", "VERY_LONG: 24h"])
    
    # Initialize service proxy helper
    service_proxy = create_proxy_helper(client, circuit_breaker_manager, redis_client)
    logger.info("Service proxy initialized with circuit breaker and caching support")
    
    yield
    
    # Shutdown
    logger.info("Shutting down aModels Gateway")
    
    if cache_manager:
        stats = cache_manager.get_stats()
        logger.info("Cache statistics at shutdown", **stats)
    
    if rate_limiter:
        await rate_limiter.stop_cleanup_task()
        logger.info("Rate limiter stopped")
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
    
    if client:
        await client.aclose()
        logger.info("HTTP client closed")


app = FastAPI(
    title="aModels Gateway",
    version="0.2.0",  # Bumped version for optimizations
    lifespan=lifespan
)

# Perplexity API endpoints - proxy to orchestration service
ORCHESTRATION_URL = os.getenv("ORCHESTRATION_URL", "http://localhost:8080")

# Check if orchestration service is available
async def check_orchestration_health() -> bool:
    """Check if orchestration service is available."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/healthz", timeout=2.0)
        return r.status_code == 200
    except:
        return False

# Cache health check result
_orchestration_available = None
_last_health_check = 0


# Helper function for circuit breaker-protected service calls
async def call_with_circuit_breaker(
    service_name: str,
    func,
    timeout: float = 30.0
):
    """
    Call a service with circuit breaker protection.
    
    Args:
        service_name: Name of the service (for circuit breaker identification)
        func: Async function to execute
        timeout: Request timeout in seconds
        
    Returns:
        Response from the service
        
    Raises:
        HTTPException: On service error or circuit breaker open
    """
    if circuit_breaker_manager is None:
        # Fallback if circuit breaker not initialized
        return await func()
    
    breaker = circuit_breaker_manager.get_breaker(
        service_name,
        CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0
        )
    )
    
    try:
        result = await breaker.call(func)
        return result
    except CircuitBreakerOpenError as e:
        logger.error(
            f"Circuit breaker open for {service_name}",
            service=service_name,
            error=str(e)
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporarily unavailable",
                "service": service_name,
                "message": str(e),
                "retry_after": 60
            }
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"{service_name} service error: {e}")

@app.post("/api/perplexity/process")
async def perplexity_process(payload: Dict[str, Any]) -> Any:
    """Process documents from Perplexity API."""
    # Check if orchestration is available
    global _orchestration_available, _last_health_check
    import time as time_module
    if time_module.time() - _last_health_check > 30:  # Check every 30 seconds
        _orchestration_available = await check_orchestration_health()
        _last_health_check = time_module.time()
    
    if not _orchestration_available:
        # Return mock response if orchestration not available
        return {
            "request_id": f"mock_{int(time.time())}",
            "status": "pending",
            "message": "Orchestration service not running. Start it with: cd services/orchestration && go run ./cmd/server/main.go",
            "status_url": "/api/perplexity/status/mock",
            "results_url": "/api/perplexity/results/mock"
        }
    
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/perplexity/process", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        # Fall back to mock if connection fails
        logger.warning(f"Orchestration service unavailable: {e}")
        _orchestration_available = False
        return {
            "request_id": f"mock_{int(time.time())}",
            "status": "pending",
            "message": f"Orchestration service error: {e}. Using mock response.",
            "status_url": "/api/perplexity/status/mock",
            "results_url": "/api/perplexity/results/mock"
        }

@app.get("/api/perplexity/status/{request_id}")
async def perplexity_status(request_id: str) -> Any:
    """Get processing status."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/perplexity/status/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/perplexity/results/{request_id}")
async def perplexity_results(request_id: str) -> Any:
    """Get processing results."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/perplexity/results/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/perplexity/results/{request_id}/intelligence")
async def perplexity_intelligence(request_id: str) -> Any:
    """Get intelligence data."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/perplexity/results/{request_id}/intelligence", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/perplexity/history")
async def perplexity_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    query: Optional[str] = None
) -> Any:
    """Get request history."""
    try:
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if query:
            params["query"] = query
        r = await client.get(f"{ORCHESTRATION_URL}/api/perplexity/history", params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/perplexity/search")
async def perplexity_search(payload: Dict[str, Any]) -> Any:
    """Search indexed documents."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/perplexity/search", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/perplexity/results/{request_id}/export")
async def perplexity_export(request_id: str, format: str = Query("json")) -> Any:
    """Export results."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/perplexity/results/{request_id}/export", params={"format": format}, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/perplexity/batch")
async def perplexity_batch(payload: Dict[str, Any]) -> Any:
    """Batch process multiple queries."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/perplexity/batch", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.delete("/api/perplexity/jobs/{request_id}")
async def perplexity_cancel_job(request_id: str) -> Any:
    """Cancel a job."""
    try:
        r = await client.delete(f"{ORCHESTRATION_URL}/api/perplexity/jobs/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/perplexity/learning/report")
async def perplexity_learning_report() -> Any:
    """Get learning report."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/perplexity/learning/report", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/perplexity/graph/{request_id}/query")
async def perplexity_graph_query(request_id: str, payload: Dict[str, Any]) -> Any:
    """Query knowledge graph."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/perplexity/graph/{request_id}/query", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/perplexity/graph/{request_id}/relationships")
async def perplexity_relationships(request_id: str) -> Any:
    """Get relationships."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/perplexity/graph/{request_id}/relationships", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/perplexity/domains/{domain}/documents")
async def perplexity_domain_documents(domain: str, limit: int = Query(50), offset: int = Query(0)) -> Any:
    """Get documents by domain."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/perplexity/domains/{domain}/documents", params={"limit": limit, "offset": offset}, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/perplexity/catalog/search")
async def perplexity_catalog_search(payload: Dict[str, Any]) -> Any:
    """Search catalog."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/perplexity/catalog/search", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

# DMS API endpoints - proxy to orchestration service (which now uses Extract service)
# These endpoints are kept for backward compatibility
@app.post("/api/dms/process")
async def dms_process(payload: Dict[str, Any]) -> Any:
    """Process documents via orchestration service (which uses Extract service)."""
    global _orchestration_available, _last_health_check
    import time as time_module
    if time_module.time() - _last_health_check > 30:
        _orchestration_available = await check_orchestration_health()
        _last_health_check = time_module.time()
    
    if not _orchestration_available:
        return {
            "request_id": f"mock_{int(time.time())}",
            "status": "pending",
            "message": "Orchestration service not running. Start it with: cd services/orchestration && go run ./cmd/server/main.go",
            "status_url": "/api/dms/status/mock",
            "results_url": "/api/dms/results/mock"
        }
    
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/dms/process", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        logger.warning(f"Orchestration service unavailable: {e}")
        _orchestration_available = False
        return {
            "request_id": f"mock_{int(time.time())}",
            "status": "pending",
            "message": f"Orchestration service error: {e}. Using mock response.",
            "status_url": "/api/dms/status/mock",
            "results_url": "/api/dms/results/mock"
        }

@app.get("/api/dms/status/{request_id}")
async def dms_status(request_id: str) -> Any:
    """Get DMS processing status."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/dms/status/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/dms/results/{request_id}")
async def dms_results(request_id: str) -> Any:
    """Get DMS processing results."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/dms/results/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/dms/results/{request_id}/intelligence")
async def dms_intelligence(request_id: str) -> Any:
    """Get DMS intelligence data."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/dms/results/{request_id}/intelligence", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/dms/history")
async def dms_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    document_id: Optional[str] = None
) -> Any:
    """Get DMS request history."""
    try:
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if document_id:
            params["document_id"] = document_id
        r = await client.get(f"{ORCHESTRATION_URL}/api/dms/history", params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/dms/search")
async def dms_search(payload: Dict[str, Any]) -> Any:
    """Search DMS indexed documents."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/dms/search", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/dms/results/{request_id}/export")
async def dms_export(request_id: str, format: str = Query("json")) -> Any:
    """Export DMS results."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/dms/results/{request_id}/export", params={"format": format}, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/dms/batch")
async def dms_batch(payload: Dict[str, Any]) -> Any:
    """Batch process multiple DMS documents."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/dms/batch", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.delete("/api/dms/jobs/{request_id}")
async def dms_cancel_job(request_id: str) -> Any:
    """Cancel a DMS job."""
    try:
        r = await client.delete(f"{ORCHESTRATION_URL}/api/dms/jobs/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/dms/graph/{request_id}/query")
async def dms_graph_query(request_id: str, payload: Dict[str, Any]) -> Any:
    """Query DMS knowledge graph."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/dms/graph/{request_id}/query", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/dms/domains/{domain}/documents")
async def dms_domain_documents(domain: str, limit: int = Query(50), offset: int = Query(0)) -> Any:
    """Get DMS documents by domain."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/dms/domains/{domain}/documents", params={"limit": limit, "offset": offset}, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/dms/catalog/search")
async def dms_catalog_search(payload: Dict[str, Any]) -> Any:
    """Search DMS catalog."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/dms/catalog/search", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/dms/documents/{document_id}")
async def dms_get_document(document_id: str) -> Any:
    """Get a specific document from Extract service (via orchestration)."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/dms/documents/{document_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

# Relational API endpoints - proxy to orchestration service
@app.post("/api/relational/process")
async def relational_process(payload: Dict[str, Any]) -> Any:
    """Process tables from relational database."""
    global _orchestration_available, _last_health_check
    import time as time_module
    if time_module.time() - _last_health_check > 30:
        _orchestration_available = await check_orchestration_health()
        _last_health_check = time_module.time()
    
    if not _orchestration_available:
        return {
            "request_id": f"mock_{int(time.time())}",
            "status": "pending",
            "message": "Orchestration service not running. Start it with: cd services/orchestration && go run ./cmd/server/main.go",
            "status_url": "/api/relational/status/mock",
            "results_url": "/api/relational/results/mock"
        }
    
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/relational/process", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        logger.warning(f"Orchestration service unavailable: {e}")
        _orchestration_available = False
        return {
            "request_id": f"mock_{int(time.time())}",
            "status": "pending",
            "message": f"Orchestration service error: {e}. Using mock response.",
            "status_url": "/api/relational/status/mock",
            "results_url": "/api/relational/results/mock"
        }

@app.get("/api/relational/status/{request_id}")
async def relational_status(request_id: str) -> Any:
    """Get relational processing status."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/relational/status/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/relational/results/{request_id}")
async def relational_results(request_id: str) -> Any:
    """Get relational processing results."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/relational/results/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/relational/results/{request_id}/intelligence")
async def relational_intelligence(request_id: str) -> Any:
    """Get relational intelligence data."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/relational/results/{request_id}/intelligence", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/relational/history")
async def relational_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    table: Optional[str] = None
) -> Any:
    """Get relational request history."""
    try:
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if table:
            params["table"] = table
        r = await client.get(f"{ORCHESTRATION_URL}/api/relational/history", params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/relational/search")
async def relational_search(payload: Dict[str, Any]) -> Any:
    """Search relational indexed tables."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/relational/search", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/relational/results/{request_id}/export")
async def relational_export(request_id: str, format: str = Query("json")) -> Any:
    """Export relational results."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/relational/results/{request_id}/export", params={"format": format}, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/relational/batch")
async def relational_batch(payload: Dict[str, Any]) -> Any:
    """Batch process multiple relational tables."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/relational/batch", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.delete("/api/relational/jobs/{request_id}")
async def relational_cancel_job(request_id: str) -> Any:
    """Cancel a relational job."""
    try:
        r = await client.delete(f"{ORCHESTRATION_URL}/api/relational/jobs/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/relational/graph/{request_id}/query")
async def relational_graph_query(request_id: str, payload: Dict[str, Any]) -> Any:
    """Query relational knowledge graph."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/relational/graph/{request_id}/query", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/relational/domains/{domain}/tables")
async def relational_domain_tables(domain: str, limit: int = Query(50), offset: int = Query(0)) -> Any:
    """Get relational tables by domain."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/relational/domains/{domain}/tables", params={"limit": limit, "offset": offset}, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.post("/api/relational/catalog/search")
async def relational_catalog_search(payload: Dict[str, Any]) -> Any:
    """Search relational catalog."""
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/relational/catalog/search", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

# Murex API endpoints - proxy to orchestration service
@app.post("/api/murex/process")
async def murex_process(payload: Dict[str, Any]) -> Any:
    """Process Murex trades."""
    global _orchestration_available, _last_health_check
    import time as time_module
    if time_module.time() - _last_health_check > 30:
        _orchestration_available = await check_orchestration_health()
        _last_health_check = time_module.time()
    
    if not _orchestration_available:
        return {
            "request_id": f"mock_{int(time.time())}",
            "status": "pending",
            "message": "Orchestration service not running. Start it with: cd services/orchestration && go run ./cmd/server/main.go",
            "status_url": "/api/murex/status/mock",
            "results_url": "/api/murex/results/mock"
        }
    
    try:
        r = await client.post(f"{ORCHESTRATION_URL}/api/murex/process", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        logger.warning(f"Orchestration service unavailable: {e}")
        _orchestration_available = False
        return {
            "request_id": f"mock_{int(time.time())}",
            "status": "pending",
            "message": f"Orchestration service error: {e}. Using mock response.",
            "status_url": "/api/murex/status/mock",
            "results_url": "/api/murex/results/mock"
        }

@app.get("/api/murex/status/{request_id}")
async def murex_status(request_id: str) -> Any:
    """Get Murex processing status."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/murex/status/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/murex/results/{request_id}")
async def murex_results(request_id: str) -> Any:
    """Get Murex processing results."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/murex/results/{request_id}", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/murex/results/{request_id}/intelligence")
async def murex_intelligence(request_id: str) -> Any:
    """Get Murex intelligence data."""
    try:
        r = await client.get(f"{ORCHESTRATION_URL}/api/murex/results/{request_id}/intelligence", timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

@app.get("/api/murex/history")
async def murex_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    table: Optional[str] = None
) -> Any:
    """Get Murex request history."""
    try:
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if table:
            params["table"] = table
        r = await client.get(f"{ORCHESTRATION_URL}/api/murex/history", params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for correlation ID and request logging
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Add correlation ID to each request for tracing."""
    # Get or generate correlation ID
    corr_id = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID") or str(uuid.uuid4())
    set_correlation_id(corr_id)
    
    # Log request
    start_time = time.time()
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    # Add correlation ID to response headers
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    
    # Log response and record metrics
    duration = time.time() - start_time
    endpoint = request.url.path
    request_counter.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()
    request_duration.labels(
        method=request.method,
        endpoint=endpoint
    ).observe(duration)
    
    logger.info(
        f"Request completed: {request.method} {request.url.path}",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=f"{duration * 1000:.2f}"
    )
    
    return response


# Middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to requests."""
    if rate_limiter is None:
        return await call_next(request)
    
    # Skip rate limiting for health and metrics endpoints
    if request.url.path in ["/healthz", "/health", "/metrics", "/circuit-breakers"]:
        return await call_next(request)
    
    # Use client IP as identifier
    client_id = request.client.host if request.client else "unknown"
    
    # Check rate limit
    allowed, info = await rate_limiter.is_allowed(client_id, request.url.path)
    
    if not allowed:
        rate_limit_rejections.labels(endpoint=request.url.path).inc()
        logger.warning(
            "Rate limit exceeded",
            client_ip=client_id,
            path=request.url.path,
            retry_after=info["retry_after"]
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": info["retry_after"],
                "message": f"Please wait {info['retry_after']} seconds before retrying"
            },
            headers={"Retry-After": str(info["retry_after"])}
        )
    
    # Add rate limit info to response headers
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(info["tokens_remaining"])
    
    return response

REPO_ROOT = Path(__file__).resolve().parents[2]
SGMI_JSON_PATH = REPO_ROOT / "data" / "training" / "sgmi" / "json_with_changes.json"
TRAINING_DATASET_DIR = REPO_ROOT / "training_data" / "dataset"
TRAINING_FEATURES_PATH = TRAINING_DATASET_DIR / "features.json"
TRAINING_METADATA_PATH = TRAINING_DATASET_DIR / "metadata.json"


def _load_json_file(path: Path) -> Any:
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"{path} not found") from exc
    if not payload.strip():
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON in {path.name}: {exc}"
        ) from exc


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    statuses: Dict[str, Any] = {
        "gateway": "ok",
        "timestamp": time.time(),
        "version": "0.2.0"
    }
    
    # Add circuit breaker status
    if circuit_breaker_manager:
        cb_stats = circuit_breaker_manager.get_all_stats()
        statuses["circuit_breakers"] = {
            name: stats["state"] for name, stats in cb_stats.items()
        }
    
    # HANA health (with circuit breaker)
    try:
        async def check_hana():
            return await client.get(f"{HANA_URL}/healthz", timeout=5.0)
        r = await call_with_circuit_breaker("hana", check_hana, timeout=5.0)
        statuses["hana"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except HTTPException as e:
        statuses["hana"] = f"circuit_breaker_open" if e.status_code == 503 else f"error:{e.detail}"
    except Exception as e:
        statuses["hana"] = f"error:{e}"
    # AgentFlow health
    try:
        r = await client.get(f"{AGENTFLOW_URL}/healthz")
        statuses["agentflow"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["agentflow"] = f"error:{e}"
    # Extract health
    try:
        r = await client.get(f"{EXTRACT_URL}/healthz")
        statuses["extract"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["extract"] = f"error:{e}"
    # Data health
    try:
        r = await client.get(f"{DATA_URL}/healthz")
        statuses["data"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["data"] = f"error:{e}"
    # OpenSearch health
    try:
        r = await client.get(f"{OPENSEARCH_URL}")
        statuses["opensearch"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["opensearch"] = f"error:{e}"
    # Redis health
    try:
        if redis_client is None:
            raise RuntimeError("redis not initialized")
        pong = await redis_client.ping()
        statuses["redis"] = "ok" if pong else "error:pong=false"
    except Exception as e:
        statuses["redis"] = f"error:{e}"
    # LocalAI health (OpenAI-compatible /models)
    try:
        r = await client.get(f"{LOCALAI_URL}/v1/models")
        statuses["localai"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["localai"] = f"error:{e}"
    # Layer4 Browser health (if service exposes /healthz)
    try:
        r = await client.get(f"{BROWSER_URL}/healthz")
        statuses["layer4_browser"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["layer4_browser"] = f"error:{e}"
    # DeepAgents health
    try:
        r = await client.get(f"{DEEPAGENTS_URL}/healthz")
        statuses["deepagents"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["deepagents"] = f"error:{e}"
    # SAP BDC health
    try:
        r = await client.get(f"{SAP_BDC_URL}/healthz")
        statuses["sap_bdc"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["sap_bdc"] = f"error:{e}"
    # Catalog health
    try:
        r = await client.get(f"{CATALOG_URL}/healthz")
        statuses["catalog"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["catalog"] = f"error:{e}"
    # Deep Research health
    try:
        r = await client.get(f"{DEEP_RESEARCH_URL}/healthz")
        statuses["deep-research"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
    except Exception as e:
        statuses["deep-research"] = f"error:{e}"
    return statuses


@app.post("/hana/sql")
async def hana_sql(payload: Dict[str, Any]) -> Any:
    try:
        r = await client.post(f"{HANA_URL}/sql", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"HANA service error: {e}")


@app.post("/agentflow/run")
async def agentflow_run(payload: Dict[str, Any]) -> Any:
    """
    Run an AgentFlow flow via LangFlow (external service).
    This is a proxy to the AgentFlow service which manages LangFlow flows.
    """
    try:
        # AgentFlow service manages flow execution via LangFlow
        r = await client.post(f"{AGENTFLOW_URL}/flows/{payload.get('flow_id', '')}/run", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"AgentFlow service error: {e}")


@app.post("/agentflow/process")
async def agentflow_process(payload: Dict[str, Any]) -> Any:
    """
    Process AgentFlow flows via LangGraph workflow orchestration.
    This integrates AgentFlow with knowledge graphs and quality-based routing.
    """
    try:
        # Proxy to graph service for LangGraph workflow orchestration
        r = await client.post(f"{GRAPH_SERVICE_URL}/agentflow/process", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Graph service error: {e}")


@app.post("/orchestration/process")
async def orchestration_process(payload: Dict[str, Any]) -> Any:
    """
    Process orchestration chains via LangGraph workflow orchestration.
    This integrates orchestration chains with knowledge graphs and quality-based routing.
    Orchestration = Go-native LangChain-like framework for LLM chains.
    """
    try:
        # Proxy to graph service for LangGraph workflow orchestration
        r = await client.post(f"{GRAPH_SERVICE_URL}/orchestration/process", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Graph service error: {e}")


@app.post("/unified/process")
async def unified_process(payload: Dict[str, Any]) -> Any:
    """
    Process unified workflow combining knowledge graphs, orchestration chains, and AgentFlow flows.
    This is the ultimate integration endpoint that uses all three systems together.
    
    Request format:
    {
        "unified_request": {
            "knowledge_graph_request": {...},  // Optional
            "orchestration_request": {...},    // Optional
            "agentflow_request": {...},         // Optional
            "workflow_mode": "sequential"       // Optional: "sequential", "parallel", "conditional"
        }
    }
    """
    try:
        # Proxy to graph service for unified workflow orchestration
        r = await client.post(f"{GRAPH_SERVICE_URL}/unified/process", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Graph service error: {e}")


@app.post("/extract/ocr")
async def extract_ocr(payload: Dict[str, Any]) -> Any:
    try:
        r = await client.post(f"{EXTRACT_URL}/ocr", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Extract service error: {e}")


@app.post("/extract/schema-replication")
async def extract_schema_replication(payload: Dict[str, Any]) -> Any:
    try:
        r = await client.post(f"{EXTRACT_URL}/schema-replication", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Extract service error: {e}")


@app.post("/knowledge-graph/query")
async def knowledge_graph_query(payload: Dict[str, Any]) -> Any:
    """
    Execute a Cypher query against the Neo4j knowledge graph.
    
    Request format:
    {
        "query": "MATCH (n:Node) RETURN n LIMIT 10",
        "params": {"project_id": "optional", "system_id": "optional"}
    }
    """
    try:
        r = await client.post(f"{EXTRACT_URL}/knowledge-graph/query", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Extract service error: {e}")


@app.post("/deepagents/invoke")
async def deepagents_invoke(payload: Dict[str, Any]) -> Any:
    """
    Invoke the deep agent service.
    
    Request format:
    {
        "messages": [{"role": "user", "content": "..."}],
        "stream": false,
        "config": {}
    }
    """
    try:
        r = await client.post(f"{DEEPAGENTS_URL}/invoke", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"DeepAgents service error: {e}")


@app.post("/deepagents/stream")
async def deepagents_stream(payload: Dict[str, Any]):
    """
    Stream deep agent responses (Server-Sent Events).
    
    Request format:
    {
        "messages": [{"role": "user", "content": "..."}],
        "stream": true,
        "config": {}
    }
    """
    try:
        from fastapi.responses import StreamingResponse
        
        async def generate():
            async with httpx.AsyncClient(timeout=300.0) as stream_client:
                async with stream_client.stream(
                    "POST",
                    f"{DEEPAGENTS_URL}/stream",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            yield f"{line}\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"DeepAgents service error: {e}")


@app.get("/deepagents/info")
async def deepagents_info() -> Any:
    """Get information about the deep agent."""
    try:
        r = await client.get(f"{DEEPAGENTS_URL}/agent/info")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"DeepAgents service error: {e}")


@app.post("/pipeline/to-agentflow")
async def pipeline_to_agentflow(payload: Dict[str, Any]) -> Any:
    """
    Convert Control-M → SQL → Tables pipeline from knowledge graph into LangFlow flow.
    
    Request format:
    {
        "project_id": "project-123",
        "system_id": "system-456",
        "flow_name": "SGMI Pipeline",
        "flow_id": "sgmi_pipeline",
        "force": false
    }
    
    This endpoint:
    1. Queries Neo4j for Control-M jobs and their SQL/table relationships
    2. Generates a LangFlow flow JSON with logical agents
    3. Imports the flow into AgentFlow/LangFlow service
    """
    try:
        r = await client.post(f"{GRAPH_SERVICE_URL}/pipeline/to-agentflow", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Graph service error: {e}")


@app.post("/data/sql")
async def data_sql(payload: Dict[str, Any]) -> Any:
    try:
        r = await client.post(f"{DATA_URL}/sql", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Data service error: {e}")


@app.post("/search/_search")
async def opensearch_search(payload: Dict[str, Any]) -> Any:
    """OpenSearch/Elasticsearch search endpoint."""
    try:
        r = await client.post(f"{OPENSEARCH_URL}/_search", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"OpenSearch error: {e}")


def _process_results_with_stdlib(results: list, operations: list = None) -> list:
    """
    Stdlib utilities for result processing.
    Operations: deduplicate, sort_by_score, filter_by_source, truncate_content
    """
    if not operations:
        operations = ["deduplicate", "sort_by_score"]
    
    processed = list(results)
    
    # Deduplicate by id
    if "deduplicate" in operations:
        seen_ids = set()
        unique_results = []
        for result in processed:
            result_id = result.get("id", "") + result.get("source", "")
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        processed = unique_results
    
    # Sort by score (already done, but ensure it's correct)
    if "sort_by_score" in operations:
        processed.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    # Filter by source
    if "filter_by_source" in operations:
        # This would be applied if source filter is provided
        pass
    
    # Truncate content
    if "truncate_content" in operations:
        max_length = 500
        for result in processed:
            content = result.get("content", "")
            if len(content) > max_length:
                result["content"] = content[:max_length] + "..."
                result["content_truncated"] = True
    
    return processed


async def _enrich_query_with_framework(query: str) -> Dict[str, Any]:
    """
    Use framework (via graph service orchestration) to understand query intent.
    """
    try:
        # Use summarization chain to understand query intent
        orchestration_payload = {
            "orchestration_request": {
                "chain_name": "summarization",
                "inputs": {
                    "text": f"Analyze this search query and extract key entities and intent: {query}"
                }
            }
        }
        r = await client.post(f"{GRAPH_SERVICE_URL}/orchestration/process", json=orchestration_payload, timeout=5.0)
        if r.status_code == 200:
            result = r.json()
            summary = result.get("orchestration_text", "")
            return {
                "original_query": query,
                "enriched_query": query,  # Could be expanded based on summary
                "intent_summary": summary,
                "entities": [],  # Could extract entities from summary
                "enriched": True
            }
    except Exception as e:
        logger.warning(f"Framework query enrichment error: {e}")
    
    return {
        "original_query": query,
        "enriched_query": query,
        "enriched": False
    }


async def _enrich_results_with_framework(results: list, query: str) -> Dict[str, Any]:
    """
    Use framework to enrich search results with summaries and insights.
    """
    try:
        # Combine top results for summarization
        top_results = results[:5]
        combined_content = "\n\n".join([r.get("content", "")[:200] for r in top_results])
        
        orchestration_payload = {
            "orchestration_request": {
                "chain_name": "summarization",
                "inputs": {
                    "text": f"Query: {query}\n\nResults:\n{combined_content}\n\nProvide a brief summary of these search results:"
                }
            }
        }
        r = await client.post(f"{GRAPH_SERVICE_URL}/orchestration/process", json=orchestration_payload, timeout=10.0)
        if r.status_code == 200:
            result = r.json()
            summary = result.get("orchestration_text", "")
            return {
                "summary": summary,
                "insights": [],  # Could extract insights
                "enriched": True
            }
    except Exception as e:
        logger.warning(f"Framework result enrichment error: {e}")
    
    return {"enriched": False}


async def _format_results_for_prompt(results: list, max_results: int = 10) -> str:
    """
    Format search results for use in LLM prompts.
    """
    formatted = []
    for i, result in enumerate(results[:max_results]):
        formatted.append(
            f"Result {i+1}:\n"
            f"  Source: {result.get('source', 'unknown')}\n"
            f"  Score: {result.get('score', 0.0):.3f}\n"
            f"  Content: {result.get('content', '')[:300]}...\n"
        )
    return "\n".join(formatted)


async def _generate_dashboard_with_framework(
    search_results: Dict[str, Any],
    query: str
) -> Dict[str, Any]:
    """
    Use framework to generate dashboard specification from search results.
    """
    try:
        results_summary = await _format_results_for_prompt(search_results["combined_results"])
        viz_data = search_results.get("visualization", {})
        
        orchestration_payload = {
            "orchestration_request": {
                "chain_name": "dashboard_generator",
                "inputs": {
                    "query": query,
                    "search_results": results_summary,
                    "visualization_data": json.dumps(viz_data) if viz_data else "{}",
                    "metadata": json.dumps(search_results.get("metadata", {}))
                }
            }
        }
        
        r = await client.post(
            f"{GRAPH_SERVICE_URL}/orchestration/process",
            json=orchestration_payload,
            timeout=15.0
        )
        
        if r.status_code == 200:
            result = r.json()
            dashboard_text = result.get("orchestration_text", "")
            # Try to parse JSON from response
            try:
                # Extract JSON from markdown code blocks if present
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', dashboard_text, re.DOTALL)
                if json_match:
                    dashboard_text = json_match.group(1)
                else:
                    json_match = re.search(r'\{.*\}', dashboard_text, re.DOTALL)
                    if json_match:
                        dashboard_text = json_match.group(0)
                
                dashboard_spec = json.loads(dashboard_text)
                return {
                    "specification": dashboard_spec,
                    "enriched": True
                }
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse dashboard JSON from LLM response: {dashboard_text[:200]}")
                return {"enriched": False, "error": "Failed to parse dashboard specification"}
    except Exception as e:
        logger.warning(f"Framework dashboard generation error: {e}")
    
    return {"enriched": False}


async def _generate_narrative_with_framework(
    search_results: Dict[str, Any],
    query: str,
    dashboard_spec: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Use framework to generate narrative from search results.
    """
    try:
        results_summary = await _format_results_for_prompt(search_results["combined_results"], max_results=10)
        dashboard_insights = []
        if dashboard_spec and dashboard_spec.get("enriched"):
            dashboard_insights = dashboard_spec.get("specification", {}).get("insights", [])
        
        key_findings = []
        for result in search_results["combined_results"][:5]:
            if result.get("score", 0.0) > 0.7:  # High relevance results
                key_findings.append(f"- {result.get('content', '')[:100]}...")
        
        orchestration_payload = {
            "orchestration_request": {
                "chain_name": "narrative_generator",
                "inputs": {
                    "query": query,
                    "search_results_summary": results_summary,
                    "dashboard_insights": "\n".join(dashboard_insights) if dashboard_insights else "No dashboard insights available",
                    "key_findings": "\n".join(key_findings) if key_findings else "No key findings identified"
                }
            }
        }
        
        r = await client.post(
            f"{GRAPH_SERVICE_URL}/orchestration/process",
            json=orchestration_payload,
            timeout=15.0
        )
        
        if r.status_code == 200:
            result = r.json()
            narrative_text = result.get("orchestration_text", "")
            return {
                "markdown": narrative_text,
                "sections": _parse_narrative_sections(narrative_text),
                "enriched": True
            }
    except Exception as e:
        logger.warning(f"Framework narrative generation error: {e}")
    
    return {"enriched": False}


def _parse_narrative_sections(narrative_text: str) -> Dict[str, str]:
    """
    Parse narrative into sections (executive summary, findings, etc.)
    """
    sections = {}
    current_section = None
    current_content = []
    
    for line in narrative_text.split("\n"):
        # Check for markdown headers
        if line.startswith("#"):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = line.lstrip("#").strip().lower().replace(" ", "_")
            current_content = []
        else:
            current_content.append(line)
    
    if current_section:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections


async def _generate_visualization_data(results: list) -> Dict[str, Any]:
    """
    Generate data for visualization (plot service).
    Returns structured data that can be used to create charts.
    """
    # Source distribution
    source_counts = {}
    for result in results:
        source = result.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Score distribution
    scores = [r.get("score", 0.0) for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # Timeline (if metadata has timestamps)
    timeline = []
    for result in results:
        metadata = result.get("metadata", {})
        if "timestamp" in metadata:
            timeline.append({
                "timestamp": metadata["timestamp"],
                "score": result.get("score", 0.0),
                "source": result.get("source", "")
            })
    
    return {
        "source_distribution": source_counts,
        "score_statistics": {
            "average": round(avg_score, 3),
            "min": round(min(scores), 3) if scores else 0.0,
            "max": round(max(scores), 3) if scores else 0.0,
            "count": len(scores)
        },
        "timeline": timeline,
        "total_results": len(results)
    }


@app.post("/search/unified")
async def unified_search(payload: Dict[str, Any]) -> Any:
    """
    Unified search endpoint that combines multiple search backends:
    - Search Inference Service (semantic search)
    - Knowledge Graph Search (extract service)
    - Catalog Semantic Search
    - Perplexity AI (if API key configured)
    
    Request format:
    {
        "query": "search text",
        "top_k": 10,
        "sources": ["inference", "knowledge_graph", "catalog", "perplexity"],  // Optional: which sources to use
        "use_perplexity": true  // Optional: enable Perplexity web search
    }
    """
    import time
    start_time = time.time()
    
    # Request validation
    query = payload.get("query", "").strip()
    top_k = payload.get("top_k", 10)
    sources = payload.get("sources", ["inference", "knowledge_graph", "catalog"])
    use_perplexity = payload.get("use_perplexity", False) and PERPLEXITY_API_KEY != ""
    
    # Enhanced features (framework, plot, stdlib, runtime)
    enable_framework = payload.get("enable_framework", False)  # Query understanding and result enrichment
    enable_plot = payload.get("enable_plot", False)  # Visualization data
    enable_stdlib = payload.get("enable_stdlib", True)  # Result processing (default enabled)
    stdlib_operations = payload.get("stdlib_operations", ["deduplicate", "sort_by_score"])
    enable_dashboard = payload.get("enable_dashboard", False)  # Generate dynamic dashboard
    enable_narrative = payload.get("enable_narrative", False)  # Generate narrative report
    
    # Validate query
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="query too long (max 1000 characters)")
    
    # Validate top_k
    if not isinstance(top_k, int) or top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer")
    if top_k > 100:
        raise HTTPException(status_code=400, detail="top_k too large (max 100)")
    
    # Validate sources
    valid_sources = ["inference", "knowledge_graph", "catalog", "perplexity"]
    if not isinstance(sources, list):
        raise HTTPException(status_code=400, detail="sources must be a list")
    sources = [s for s in sources if s in valid_sources]
    if not sources:
        sources = ["inference", "knowledge_graph", "catalog"]
    
    # Framework: Query understanding (optional)
    query_enrichment = {}
    if enable_framework:
        query_enrichment = await _enrich_query_with_framework(query)
        # Use enriched query if available
        if query_enrichment.get("enriched_query") and query_enrichment.get("enriched"):
            # Could use enriched query, but for now keep original
            pass
    
    results = {
        "query": query,
        "sources": {},
        "combined_results": [],
        "total_count": 0,
        "metadata": {
            "sources_queried": len(sources),
            "sources_successful": 0,
            "sources_failed": 0
        }
    }
    
    # Add query enrichment if enabled
    if enable_framework and query_enrichment:
        results["query_enrichment"] = query_enrichment
    
    # 1. Search Inference Service
    if "inference" in sources:
        try:
            search_payload = {"query": query, "top_k": top_k}
            # Use retry logic for transient connection errors
            r = await retry_http_request(
                client,
                "POST",
                f"{SEARCH_INFERENCE_URL}/v1/search",
                max_retries=2,  # 2 retries = 3 total attempts
                initial_delay=0.5,
                json=search_payload,
                timeout=10.0
            )
            if r.status_code == 200:
                inference_results = r.json()
                results["sources"]["inference"] = inference_results.get("results", [])
                results["metadata"]["sources_successful"] += 1
                for result in inference_results.get("results", []):
                    results["combined_results"].append({
                        "source": "inference",
                        "id": result.get("id", ""),
                        "content": result.get("content", ""),
                        "similarity": result.get("similarity", 0.0),
                        "score": result.get("similarity", 0.0)
                    })
            else:
                raise Exception(f"HTTP {r.status_code}: {r.text[:200]}")
        except httpx.ConnectError as e:
            error_msg = f"Connection refused: {SEARCH_INFERENCE_URL} - Service may not be running (after retries)"
            logger.warning(f"Search inference service connection error: {error_msg}")
            results["sources"]["inference"] = {"error": error_msg, "url": SEARCH_INFERENCE_URL, "type": "connection_error"}
            results["metadata"]["sources_failed"] += 1
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {SEARCH_INFERENCE_URL} - Service may be overloaded (after retries)"
            logger.warning(f"Search inference service timeout: {error_msg}")
            results["sources"]["inference"] = {"error": error_msg, "url": SEARCH_INFERENCE_URL, "type": "timeout"}
            results["metadata"]["sources_failed"] += 1
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Search inference service error: {error_msg}")
            results["sources"]["inference"] = {"error": error_msg, "url": SEARCH_INFERENCE_URL, "type": "unknown_error"}
            results["metadata"]["sources_failed"] += 1
    
    # 2. Knowledge Graph Search (Extract Service)
    if "knowledge_graph" in sources:
        try:
            kg_payload = {
                "query": query,
                "limit": top_k,
                "use_semantic": True,
                "use_hybrid_search": True
            }
            # Use retry logic for transient connection errors
            r = await retry_http_request(
                client,
                "POST",
                f"{EXTRACT_URL}/knowledge-graph/search",
                max_retries=2,
                initial_delay=0.5,
                json=kg_payload,
                timeout=10.0
            )
            if r.status_code == 200:
                kg_results = r.json()
                results["sources"]["knowledge_graph"] = kg_results.get("results", [])
                results["metadata"]["sources_successful"] += 1
                for result in kg_results.get("results", []):
                    results["combined_results"].append({
                        "source": "knowledge_graph",
                        "id": result.get("id", ""),
                        "content": result.get("content") or result.get("text", ""),
                        "similarity": result.get("score", 0.0),
                        "score": result.get("score", 0.0),
                        "metadata": result.get("metadata", {})
                    })
            else:
                raise Exception(f"HTTP {r.status_code}: {r.text[:200]}")
        except httpx.ConnectError as e:
            error_msg = f"Connection refused: {EXTRACT_URL} - Service may not be running (after retries)"
            logger.warning(f"Knowledge graph search connection error: {error_msg}")
            results["sources"]["knowledge_graph"] = {"error": error_msg, "url": EXTRACT_URL, "type": "connection_error"}
            results["metadata"]["sources_failed"] += 1
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {EXTRACT_URL} - Service may be overloaded (after retries)"
            logger.warning(f"Knowledge graph search timeout: {error_msg}")
            results["sources"]["knowledge_graph"] = {"error": error_msg, "url": EXTRACT_URL, "type": "timeout"}
            results["metadata"]["sources_failed"] += 1
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Knowledge graph search error: {error_msg}")
            results["sources"]["knowledge_graph"] = {"error": error_msg, "url": EXTRACT_URL, "type": "unknown_error"}
            results["metadata"]["sources_failed"] += 1
    
    # 3. Catalog Semantic Search
    if "catalog" in sources:
        try:
            catalog_payload = {"query": query, "limit": top_k}
            # Use retry logic for transient connection errors
            r = await retry_http_request(
                client,
                "POST",
                f"{CATALOG_URL}/catalog/semantic-search",
                max_retries=2,
                initial_delay=0.5,
                json=catalog_payload,
                timeout=10.0
            )
            if r.status_code == 200:
                catalog_results = r.json()
                results["sources"]["catalog"] = catalog_results.get("results", [])
                results["metadata"]["sources_successful"] += 1
                for result in catalog_results.get("results", []):
                    results["combined_results"].append({
                        "source": "catalog",
                        "id": result.get("id", ""),
                        "content": result.get("content") or result.get("text", ""),
                        "similarity": result.get("score", 0.0),
                        "score": result.get("score", 0.0),
                        "metadata": result.get("metadata", {})
                    })
            else:
                raise Exception(f"HTTP {r.status_code}: {r.text[:200]}")
        except httpx.ConnectError as e:
            error_msg = f"Connection refused: {CATALOG_URL} - Service may not be running (after retries)"
            logger.warning(f"Catalog search connection error: {error_msg}")
            results["sources"]["catalog"] = {"error": error_msg, "url": CATALOG_URL, "type": "connection_error"}
            results["metadata"]["sources_failed"] += 1
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {CATALOG_URL} - Service may be overloaded (after retries)"
            logger.warning(f"Catalog search timeout: {error_msg}")
            results["sources"]["catalog"] = {"error": error_msg, "url": CATALOG_URL, "type": "timeout"}
            results["metadata"]["sources_failed"] += 1
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Catalog search error: {error_msg}")
            results["sources"]["catalog"] = {"error": error_msg, "url": CATALOG_URL, "type": "unknown_error"}
            results["metadata"]["sources_failed"] += 1
    
    # 4. Perplexity AI (web search)
    if use_perplexity and PERPLEXITY_API_KEY:
        try:
            perplexity_payload = {
                "model": "sonar",
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 500
            }
            r = await client.post(
                "https://api.perplexity.ai/chat/completions",
                json=perplexity_payload,
                headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}"},
                timeout=30.0
            )
            if r.status_code == 200:
                perplexity_response = r.json()
                content = perplexity_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                citations = perplexity_response.get("citations", [])
                results["sources"]["perplexity"] = {
                    "content": content,
                    "citations": citations
                }
                results["metadata"]["sources_successful"] += 1
                results["combined_results"].append({
                    "source": "perplexity",
                    "id": "perplexity-web",
                    "content": content,
                    "similarity": 1.0,  # Perplexity results are always relevant
                    "score": 1.0,
                    "citations": citations,
                    "metadata": {"type": "web_search"}
                })
            else:
                raise Exception(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"Perplexity search error: {e}")
            results["sources"]["perplexity"] = {"error": str(e)}
            results["metadata"]["sources_failed"] += 1
    
    # Stdlib: Process results (deduplicate, sort, truncate)
    if enable_stdlib:
        results["combined_results"] = _process_results_with_stdlib(
            results["combined_results"],
            operations=stdlib_operations
        )
    else:
        # Just sort if stdlib is disabled
        results["combined_results"].sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    # Limit to top_k
    results["combined_results"] = results["combined_results"][:top_k]
    results["total_count"] = len(results["combined_results"])
    
    # Framework: Result enrichment (optional)
    if enable_framework and results["combined_results"]:
        result_enrichment = await _enrich_results_with_framework(
            results["combined_results"],
            query
        )
        if result_enrichment.get("enriched"):
            results["result_enrichment"] = result_enrichment
    
    # Plot: Generate visualization data (optional)
    if enable_plot and results["combined_results"]:
        visualization_data = await _generate_visualization_data(results["combined_results"])
        results["visualization"] = visualization_data
    
    # Framework: Generate dynamic dashboard (optional)
    dashboard_spec = None
    if enable_dashboard and results["combined_results"]:
        dashboard_spec = await _generate_dashboard_with_framework(results, query)
        if dashboard_spec.get("enriched"):
            results["dashboard"] = dashboard_spec
    
    # Framework: Generate narrative report (optional)
    if enable_narrative and results["combined_results"]:
        narrative = await _generate_narrative_with_framework(
            results,
            query,
            dashboard_spec=dashboard_spec
        )
        if narrative.get("enriched"):
            results["narrative"] = narrative
    
    # Add execution time to metadata
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    results["metadata"]["execution_time_ms"] = round(execution_time, 2)
    
    return results


@app.post("/search/narrative")
async def generate_search_narrative(
    payload: Dict[str, Any],
    stream: bool = False
) -> Any:
    """
    Generate narrative from search results using framework.
    
    Request:
    {
        "query": "search query",
        "search_results": {...},  // Optional: if not provided, performs search first
        "enable_framework": true
    }
    """
    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    
    search_results = payload.get("search_results")
    
    # If search results not provided, perform search first
    if not search_results:
        search_payload = {
            "query": query,
            "enable_framework": True,
            "enable_plot": True
        }
        search_response = await unified_search(search_payload)
        search_results = search_response
    
    # Generate narrative
    if stream:
        # Streaming mode
        try:
            from streaming_utils import stream_narrative_generation
            
            # Make streaming request to orchestration service
            results_summary = await _format_results_for_prompt(search_results["combined_results"], max_results=10)
            key_findings = []
            for result in search_results["combined_results"][:5]:
                if result.get("score", 0.0) > 0.7:
                    key_findings.append(f"- {result.get('content', '')[:100]}...")
            
            orchestration_payload = {
                "orchestration_request": {
                    "chain_name": "narrative_generator",
                    "inputs": {
                        "query": query,
                        "search_results_summary": results_summary,
                        "dashboard_insights": "No dashboard insights available",
                        "key_findings": "\n".join(key_findings) if key_findings else "No key findings identified"
                    }
                }
            }
            
            # Note: This requires the orchestration service to support streaming
            # For now, we'll return a non-streaming response
            # In production, you'd use: r = await client.post(..., stream=True)
            narrative = await _generate_narrative_with_framework(search_results, query)
            
            return Response(
                content=json.dumps({
                    "query": query,
                    "narrative": narrative,
                    "search_metadata": search_results.get("metadata", {})
                }),
                media_type="application/json"
            )
        except Exception as e:
            logger.warning(f"Streaming not available, falling back to regular response: {e}")
            narrative = await _generate_narrative_with_framework(search_results, query)
    else:
        narrative = await _generate_narrative_with_framework(search_results, query)
    
    return {
        "query": query,
        "narrative": narrative,
        "search_metadata": search_results.get("metadata", {})
    }


@app.post("/search/dashboard")
async def generate_search_dashboard(payload: Dict[str, Any]) -> Any:
    """
    Generate dashboard configuration from search results using framework.
    
    Request:
    {
        "query": "search query",
        "search_results": {...},  // Optional: if not provided, performs search first
        "enable_framework": true,
        "enable_plot": true
    }
    """
    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    
    search_results = payload.get("search_results")
    
    # If search results not provided, perform search first
    if not search_results:
        search_payload = {
            "query": query,
            "enable_framework": True,
            "enable_plot": True
        }
        search_response = await unified_search(search_payload)
        search_results = search_response
    
    # Generate dashboard
    dashboard = await _generate_dashboard_with_framework(search_results, query)
    
    return {
        "query": query,
        "dashboard": dashboard,
        "search_metadata": search_results.get("metadata", {})
    }


@app.get("/search/export/narrative/{query_hash}")
async def export_narrative_to_powerpoint(
    query_hash: str,
    narrative_data: Dict[str, Any] = None
) -> Any:
    """
    Export narrative to PowerPoint format.
    
    Note: This is a simplified endpoint. In production, you'd store the narrative
    data and retrieve it by hash, or pass it in the request body.
    """
    try:
        from export_powerpoint import create_powerpoint_from_narrative
        
        # For now, we'll need the narrative data in the request
        # In production, this would be stored and retrieved by hash
        if not narrative_data:
            raise HTTPException(status_code=400, detail="Narrative data required")
        
        pptx_file = create_powerpoint_from_narrative(
            narrative_data.get("narrative", {}),
            narrative_data.get("query", ""),
            narrative_data.get("search_metadata", {})
        )
        
        return StreamingResponse(
            pptx_file,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={
                "Content-Disposition": f'attachment; filename="narrative_{query_hash}.pptx"'
            }
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PowerPoint export requires python-pptx. Install with: pip install python-pptx"
        )
    except Exception as e:
        logger.error(f"PowerPoint export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/export/narrative")
async def export_narrative_to_powerpoint_post(payload: Dict[str, Any]) -> Any:
    """
    Export narrative to PowerPoint format.
    
    Request:
    {
        "query": "search query",
        "narrative": {...},
        "search_metadata": {...}
    }
    """
    try:
        from export_powerpoint import create_powerpoint_from_narrative
        
        query = payload.get("query", "search_results")
        narrative = payload.get("narrative", {})
        search_metadata = payload.get("search_metadata", {})
        
        if not narrative.get("enriched"):
            raise HTTPException(status_code=400, detail="Narrative not available or not enriched")
        
        pptx_file = create_powerpoint_from_narrative(
            narrative,
            query,
            search_metadata
        )
        
        # Generate filename from query
        import re
        filename = re.sub(r'[^\w\s-]', '', query)[:50].strip().replace(' ', '_')
        filename = filename or "narrative"
        
        return StreamingResponse(
            pptx_file,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}_narrative.pptx"'
            }
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PowerPoint export requires python-pptx. Install with: pip install python-pptx"
        )
    except Exception as e:
        logger.error(f"PowerPoint export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/export/dashboard")
async def export_dashboard_to_powerpoint(payload: Dict[str, Any]) -> Any:
    """
    Export dashboard to PowerPoint format.
    
    Request:
    {
        "query": "search query",
        "dashboard": {...},
        "search_metadata": {...}
    }
    """
    try:
        from export_powerpoint import create_powerpoint_from_dashboard
        
        query = payload.get("query", "search_results")
        dashboard = payload.get("dashboard", {})
        search_metadata = payload.get("search_metadata", {})
        
        if not dashboard.get("enriched"):
            raise HTTPException(status_code=400, detail="Dashboard not available or not enriched")
        
        pptx_file = create_powerpoint_from_dashboard(
            dashboard,
            query,
            search_metadata
        )
        
        # Generate filename from query
        import re
        filename = re.sub(r'[^\w\s-]', '', query)[:50].strip().replace(' ', '_')
        filename = filename or "dashboard"
        
        return StreamingResponse(
            pptx_file,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}_dashboard.pptx"'
            }
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PowerPoint export requires python-pptx. Install with: pip install python-pptx"
        )
    except Exception as e:
        logger.error(f"PowerPoint export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/export/narrative-dashboard")
async def export_narrative_and_dashboard_to_powerpoint(payload: Dict[str, Any]) -> Any:
    """
    Export combined narrative and dashboard to PowerPoint format.
    
    Request:
    {
        "query": "search query",
        "narrative": {...},
        "dashboard": {...},
        "search_metadata": {...}
    }
    """
    try:
        from export_powerpoint import create_powerpoint_from_narrative_and_dashboard
        
        query = payload.get("query", "search_results")
        narrative = payload.get("narrative", {})
        dashboard = payload.get("dashboard", {})
        search_metadata = payload.get("search_metadata", {})
        
        if not narrative.get("enriched") or not dashboard.get("enriched"):
            raise HTTPException(
                status_code=400,
                detail="Both narrative and dashboard must be available and enriched"
            )
        
        pptx_file = create_powerpoint_from_narrative_and_dashboard(
            narrative,
            dashboard,
            query,
            search_metadata
        )
        
        # Generate filename from query
        import re
        filename = re.sub(r'[^\w\s-]', '', query)[:50].strip().replace(' ', '_')
        filename = filename or "report"
        
        return StreamingResponse(
            pptx_file,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}_report.pptx"'
            }
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PowerPoint export requires python-pptx. Install with: pip install python-pptx"
        )
    except Exception as e:
        logger.error(f"PowerPoint export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/narrative-dashboard")
async def generate_narrative_and_dashboard(payload: Dict[str, Any]) -> Any:
    """
    Generate both narrative and dashboard from search results.
    
    Request:
    {
        "query": "search query",
        "search_results": {...},  // Optional
        "enable_framework": true,
        "enable_plot": true
    }
    """
    import asyncio
    
    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    
    search_results = payload.get("search_results")
    
    # If search results not provided, perform search first
    if not search_results:
        search_payload = {
            "query": query,
            "enable_framework": True,
            "enable_plot": True
        }
        search_response = await unified_search(search_payload)
        search_results = search_response
    
    # Generate both in parallel
    narrative_task = _generate_narrative_with_framework(search_results, query)
    dashboard_task = _generate_dashboard_with_framework(search_results, query)
    
    narrative, dashboard = await asyncio.gather(
        narrative_task,
        dashboard_task,
        return_exceptions=True
    )
    
    return {
        "query": query,
        "narrative": narrative if not isinstance(narrative, Exception) else {"error": str(narrative)},
        "dashboard": dashboard if not isinstance(dashboard, Exception) else {"error": str(dashboard)},
        "search_metadata": search_results.get("metadata", {})
    }


@app.post("/localai/chat")
async def localai_chat(payload: Dict[str, Any]) -> Any:
    # payload expects: { model: string, messages: [ {role, content}, ... ] }
    try:
        r = await client.post(f"{LOCALAI_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"LocalAI error: {e}")


@app.get("/browser/health")
async def browser_health() -> Any:
    try:
        r = await client.get(f"{BROWSER_URL}/healthz")
        r.raise_for_status()
        return {"ok": True}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Layer4 Browser error: {e}")


@app.get("/shell/sgmi/raw")
async def shell_sgmi_raw() -> Any:
    """Expose the Control-M hierarchy used by the browser shell."""

    return _load_json_file(SGMI_JSON_PATH)


@app.get("/shell/training/dataset")
async def shell_training_dataset() -> Dict[str, Any]:
    """Expose the derived training dataset summary for the browser shell."""

    metadata = _load_json_file(TRAINING_METADATA_PATH)
    dataset = _load_json_file(TRAINING_FEATURES_PATH)
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=500, detail="training metadata is not a JSON object")
    if not isinstance(dataset, dict):
        raise HTTPException(status_code=500, detail="training dataset is not a JSON object")
    return {"metadata": metadata, "dataset": dataset}


@app.get("/redis/get")
async def redis_get(key: str = Query(...)) -> Any:
    try:
        if redis_client is None:
            raise RuntimeError("redis not initialized")
        val = await redis_client.get(key)
        return {"key": key, "value": None if val is None else val.decode("utf-8")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redis/set")
async def redis_set(payload: Dict[str, Any]) -> Any:
    try:
        if redis_client is None:
            raise RuntimeError("redis not initialized")
        key = payload.get("key")
        value = payload.get("value")
        ex = payload.get("ex")  # seconds
        if not isinstance(key, str):
            raise HTTPException(status_code=400, detail="key required")
        if value is None:
            raise HTTPException(status_code=400, detail="value required")
        await redis_client.set(key, str(value), ex=ex)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Startup/shutdown now handled by lifespan context manager

# New observability endpoints

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/circuit-breakers")
async def circuit_breakers_status() -> Dict[str, Any]:
    """Get status of all circuit breakers."""
    if circuit_breaker_manager is None:
        return {"error": "Circuit breaker manager not initialized"}
    
    stats = circuit_breaker_manager.get_all_stats()
    
    # Update Prometheus gauges
    for service, breaker_stats in stats.items():
        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(breaker_stats["state"], -1)
        circuit_breaker_state.labels(service=service).set(state_value)
    
    return {
        "circuit_breakers": stats,
        "total_breakers": len(stats)
    }


@app.post("/circuit-breakers/reset")
async def reset_circuit_breakers() -> Dict[str, str]:
    """Manually reset all circuit breakers (admin operation)."""
    if circuit_breaker_manager is None:
        raise HTTPException(status_code=503, detail="Circuit breaker manager not initialized")
    
    await circuit_breaker_manager.reset_all()
    logger.info("All circuit breakers manually reset")
    
    return {"status": "success", "message": "All circuit breakers reset to closed state"}


@app.get("/rate-limit/stats")
async def rate_limit_stats() -> Dict[str, Any]:
    """Get rate limiter statistics."""
    if rate_limiter is None:
        return {"error": "Rate limiter not initialized"}
    
    return rate_limiter.get_stats()


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats() -> CacheStats:
    """Get cache statistics."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")
    
    stats = cache_manager.get_stats()
    return CacheStats(
        total_keys=stats["hits"] + stats["misses"],
        hit_rate=stats["hit_rate"],
        miss_rate=stats["miss_rate"],
        total_hits=stats["hits"],
        total_misses=stats["misses"]
    )


@app.post("/cache/invalidate")
async def invalidate_cache(pattern: str = Query(..., description="Redis key pattern to invalidate")) -> GenericResponse:
    """Invalidate cache entries matching pattern (admin operation)."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")
    
    deleted = await cache_manager.invalidate_pattern(pattern)
    logger.info("Cache invalidated via API", pattern=pattern, deleted_keys=deleted)
    
    return GenericResponse(
        status="success",
        message=f"Invalidated {deleted} cache entries",
        data={"pattern": pattern, "deleted_keys": deleted}
    )


@app.post("/cache/clear")
async def clear_all_cache() -> GenericResponse:
    """Clear all gateway cache entries (admin operation)."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")
    
    success = await cache_manager.clear_all()
    
    if success:
        return GenericResponse(
            status="success",
            message="All cache entries cleared"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.get("/telemetry/recent")
async def telemetry_recent() -> Any:
    try:
        r = await client.get(f"{DATA_URL}/telemetry/recent")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Data service error: {e}")


@app.post("/sap-bdc/extract")
async def sap_bdc_extract(payload: Dict[str, Any]) -> Any:
    """
    Extract data and schema from SAP Business Data Cloud.
    
    Request format:
    {
        "formation_id": "formation-123",
        "source_system": "SAP S/4HANA Cloud",
        "data_product_id": "product-456",
        "space_id": "space-789",
        "database": "HANADB",
        "include_views": true,
        "options": {}
    }
    """
    try:
        r = await client.post(f"{SAP_BDC_URL}/extract", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"SAP BDC service error: {e}")


@app.get("/sap-bdc/data-products")
async def sap_bdc_data_products() -> Any:
    """List all available data products in the formation."""
    try:
        r = await client.get(f"{SAP_BDC_URL}/data-products")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"SAP BDC service error: {e}")


@app.get("/sap-bdc/intelligent-applications")
async def sap_bdc_intelligent_applications() -> Any:
    """List all available intelligent applications."""
    try:
        r = await client.get(f"{SAP_BDC_URL}/intelligent-applications")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"SAP BDC service error: {e}")


@app.get("/sap-bdc/formation")
async def sap_bdc_formation() -> Any:
    """Get formation details including components and data sources."""
    try:
        r = await client.get(f"{SAP_BDC_URL}/formation")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"SAP BDC service error: {e}")


@app.get("/catalog/data-elements")
async def catalog_data_elements() -> Any:
    """List all data elements in the catalog."""
    try:
        r = await client.get(f"{CATALOG_URL}/catalog/data-elements")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.get("/catalog/data-elements/{element_id}")
async def catalog_get_data_element(element_id: str) -> Any:
    """Get a specific data element by ID."""
    try:
        r = await client.get(f"{CATALOG_URL}/catalog/data-elements/{element_id}")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.post("/catalog/data-elements")
async def catalog_create_data_element(payload: Dict[str, Any]) -> Any:
    """Register a new data element in the catalog."""
    try:
        r = await client.post(f"{CATALOG_URL}/catalog/data-elements", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.get("/catalog/ontology")
async def catalog_ontology() -> Any:
    """Get the OWL ontology metadata."""
    try:
        r = await client.get(f"{CATALOG_URL}/catalog/ontology")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.post("/catalog/semantic-search")
async def catalog_semantic_search(payload: Dict[str, Any]) -> Any:
    """Perform semantic search on the catalog."""
    try:
        r = await client.post(f"{CATALOG_URL}/catalog/semantic-search", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.post("/catalog/sparql")
async def catalog_sparql(payload: Dict[str, Any] = None, query: str = Query(None)) -> Any:
    """Execute a SPARQL query against the catalog triplestore."""
    try:
        # Support both POST body and GET query parameter
        if query:
            r = await client.get(f"{CATALOG_URL}/catalog/sparql", params={"query": query})
        else:
            r = await client.post(f"{CATALOG_URL}/catalog/sparql", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.get("/catalog/sparql")
async def catalog_sparql_get(query: str = Query(...)) -> Any:
    """Execute a SPARQL query via GET request."""
    try:
        r = await client.get(f"{CATALOG_URL}/catalog/sparql", params={"query": query})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.post("/catalog/data-products/build")
async def catalog_build_data_product(payload: Dict[str, Any]) -> Any:
    """
    Build a complete, end-to-end data product (thin slice approach).
    
    Request format:
    {
        "topic": "customer_data",
        "customer_need": "I need to analyze customer purchase patterns"
    }
    """
    try:
        r = await client.post(f"{CATALOG_URL}/catalog/data-products/build", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.get("/catalog/data-products/{product_id}")
async def catalog_get_data_product(product_id: str) -> Any:
    """Get a complete data product by ID."""
    try:
        r = await client.get(f"{CATALOG_URL}/catalog/data-products/{product_id}")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Catalog service error: {e}")


@app.post("/deep-research/research")
async def deep_research_research(payload: Dict[str, Any]) -> Any:
    """
    Perform deep research using Open Deep Research.
    
    Request format:
    {
        "query": "What data elements exist for customer data?",
        "context": {"topic": "customer_data"},
        "tools": ["sparql_query", "catalog_search"]
    }
    """
    try:
        r = await client.post(f"{DEEP_RESEARCH_URL}/research", json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Deep Research service error: {e}")


@app.get("/deep-research/healthz")
async def deep_research_healthz() -> Any:
    """Check Deep Research service health."""
    try:
        r = await client.get(f"{DEEP_RESEARCH_URL}/healthz")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Deep Research service error: {e}")


# ============================================================================
# REFACTORED ENDPOINTS - Examples using new DRY patterns and Pydantic models
# ============================================================================

@app.post("/v2/catalog/semantic-search", response_model=Dict[str, Any])
async def catalog_semantic_search_v2(request: SemanticSearchRequest) -> Dict[str, Any]:
    """
    Semantic search in catalog (REFACTORED VERSION with caching).
    Demonstrates: Pydantic validation, ServiceProxy, automatic caching.
    """
    if service_proxy is None:
        raise HTTPException(status_code=503, detail="Service proxy not initialized")
    
    # ServiceProxy handles circuit breaker, caching automatically
    result = await service_proxy.proxy_post(
        service_name="catalog",
        service_url=CATALOG_URL,
        path="/catalog/semantic-search",
        payload=request.dict(),
        timeout=30.0
    )
    
    return result


@app.post("/v2/catalog/data-elements", response_model=Dict[str, Any])
async def create_data_element_v2(request: DataElementRequest) -> Dict[str, Any]:
    """
    Create data element (REFACTORED VERSION with validation).
    Demonstrates: Pydantic validation for request body.
    """
    if service_proxy is None:
        raise HTTPException(status_code=503, detail="Service proxy not initialized")
    
    result = await service_proxy.proxy_post(
        service_name="catalog",
        service_url=CATALOG_URL,
        path="/catalog/data-elements",
        payload=request.dict()
    )
    
    # Invalidate catalog cache after write operation
    if cache_manager:
        await cache_manager.invalidate_pattern("gateway:cache:*:catalog*")
    
    return result


@app.get("/v2/catalog/data-elements", response_model=Dict[str, Any])
async def list_data_elements_v2() -> Dict[str, Any]:
    """
    List data elements (REFACTORED VERSION with caching).
    Demonstrates: ServiceProxy with automatic caching (LONG strategy = 1 hour).
    """
    if service_proxy is None:
        raise HTTPException(status_code=503, detail="Service proxy not initialized")
    
    # GET requests with cache_ttl enabled = automatic caching
    result = await service_proxy.proxy_get(
        service_name="catalog",
        service_url=CATALOG_URL,
        path="/catalog/data-elements",
        cache_ttl=3600  # 1 hour cache
    )
    
    return result


@app.post("/v2/deep-research/research", response_model=Dict[str, Any])
async def deep_research_v2(request: ResearchRequest) -> Dict[str, Any]:
    """
    Deep research (REFACTORED VERSION with validation and caching).
    Demonstrates: Complex Pydantic model, aggressive caching for expensive operations.
    """
    if service_proxy is None:
        raise HTTPException(status_code=503, detail="Service proxy not initialized")
    
    result = await service_proxy.proxy_post(
        service_name="deep-research",
        service_url=DEEP_RESEARCH_URL,
        path="/research",
        payload=request.dict(),
        timeout=300.0  # Long timeout for research
    )
    
    return result


@app.get("/v2/sap-bdc/data-products", response_model=Dict[str, Any])
async def sap_bdc_data_products_v2() -> Dict[str, Any]:
    """
    List SAP BDC data products (REFACTORED VERSION).
    Demonstrates: GET with long caching for relatively static data.
    """
    if service_proxy is None:
        raise HTTPException(status_code=503, detail="Service proxy not initialized")
    
    result = await service_proxy.proxy_get(
        service_name="sap-bdc",
        service_url=SAP_BDC_URL,
        path="/data-products",
        cache_ttl=3600  # 1 hour - data products don't change often
    )
    
    return result


@app.post("/v2/knowledge-graph/query", response_model=Dict[str, Any])
async def knowledge_graph_query_v2(request: GraphQueryRequest) -> Dict[str, Any]:
    """
    Query knowledge graph (REFACTORED VERSION).
    Demonstrates: Validated graph query with Pydantic model.
    """
    if service_proxy is None:
        raise HTTPException(status_code=503, detail="Service proxy not initialized")
    
    result = await service_proxy.proxy_post(
        service_name="graph",
        service_url=GRAPH_SERVICE_URL,
        path="/knowledge-graph/query",
        payload=request.dict(),
        timeout=30.0
    )
    
    return result


# ============================================================================
# Additional Admin/Debug Endpoints
# ============================================================================

@app.get("/debug/config")
async def debug_config() -> Dict[str, Any]:
    """Debug endpoint showing current configuration (admin only)."""
    return {
        "version": "0.2.0",
        "services": {
            "orchestration": ORCHESTRATION_URL,
            "hana": HANA_URL,
            "catalog": CATALOG_URL,
            "deep_research": DEEP_RESEARCH_URL,
            "graph": GRAPH_SERVICE_URL,
            "sap_bdc": SAP_BDC_URL,
        },
        "features": {
            "circuit_breakers": circuit_breaker_manager is not None,
            "rate_limiting": rate_limiter is not None,
            "caching": cache_manager is not None and redis_client is not None,
            "service_proxy": service_proxy is not None,
            "structured_logging": True,
            "prometheus_metrics": True,
        },
        "configuration": {
            "http_timeout": "60s",
            "max_connections": 100,
            "keepalive_connections": 20,
            "log_level": LOG_LEVEL,
            "json_logs": JSON_LOGS,
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
