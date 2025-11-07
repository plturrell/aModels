from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import time

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis


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

logger = logging.getLogger(__name__)
if LOCALAI_URL == GRAPH_SERVICE_URL:
    logger.warning(
        "LOCALAI_URL (%s) matches GRAPH_SERVICE_URL (%s). "
        "Override one of them to avoid routing conflicts.",
        LOCALAI_URL,
        GRAPH_SERVICE_URL,
    )

client = httpx.AsyncClient(timeout=30.0)

app = FastAPI(title="aModels Gateway", version="0.1.0")
redis_client: aioredis.Redis | None = None

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    statuses: Dict[str, Any] = {"gateway": "ok"}
    # HANA health
    try:
        r = await client.get(f"{HANA_URL}/healthz")
        statuses["hana"] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
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
            r = await client.post(f"{SEARCH_INFERENCE_URL}/v1/search", json=search_payload, timeout=10.0)
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
        except Exception as e:
            logger.warning(f"Search inference service error: {e}")
            results["sources"]["inference"] = {"error": str(e)}
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
            r = await client.post(f"{EXTRACT_URL}/knowledge-graph/search", json=kg_payload, timeout=10.0)
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
        except Exception as e:
            logger.warning(f"Knowledge graph search error: {e}")
            results["sources"]["knowledge_graph"] = {"error": str(e)}
            results["metadata"]["sources_failed"] += 1
    
    # 3. Catalog Semantic Search
    if "catalog" in sources:
        try:
            catalog_payload = {"query": query, "limit": top_k}
            r = await client.post(f"{CATALOG_URL}/catalog/semantic-search", json=catalog_payload, timeout=10.0)
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
        except Exception as e:
            logger.warning(f"Catalog search error: {e}")
            results["sources"]["catalog"] = {"error": str(e)}
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
    
    # Add execution time to metadata
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    results["metadata"]["execution_time_ms"] = round(execution_time, 2)
    
    return results


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


@app.on_event("startup")
async def on_startup() -> None:
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global redis_client
    if redis_client is not None:
        await redis_client.close()


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
