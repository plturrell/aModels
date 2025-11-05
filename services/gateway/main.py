import os
from typing import Any, Dict

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
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
BROWSER_URL = os.getenv("BROWSER_URL", "http://localhost:8070")
DEEPAGENTS_URL = os.getenv("DEEPAGENTS_URL", "http://localhost:9004")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8081")
SAP_BDC_URL = os.getenv("SAP_BDC_URL", "http://localhost:8083")
CATALOG_URL = os.getenv("CATALOG_URL", "http://localhost:8084")

client = httpx.AsyncClient(timeout=30.0)

app = FastAPI(title="aModels Gateway", version="0.1.0")
redis_client: aioredis.Redis | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        graph_url = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8081")
        r = await client.post(f"{graph_url}/agentflow/process", json=payload)
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
        graph_url = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8081")
        r = await client.post(f"{graph_url}/orchestration/process", json=payload)
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
        graph_url = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8081")
        r = await client.post(f"{graph_url}/unified/process", json=payload)
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
        graph_url = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8081")
        r = await client.post(f"{graph_url}/pipeline/to-agentflow", json=payload)
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
    try:
        r = await client.post(f"{OPENSEARCH_URL}/_search", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"OpenSearch error: {e}")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)


