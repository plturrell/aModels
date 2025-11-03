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
    try:
        r = await client.post(f"{AGENTFLOW_URL}/run", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"AgentFlow service error: {e}")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)


