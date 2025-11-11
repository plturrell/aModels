import os
import json
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # shim will still work without redis

LOCALAI_CORE = os.getenv("LOCALAI_CORE_URL", "http://localai:8080")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_KEY = os.getenv("REDIS_DOMAIN_CONFIG_KEY", "localai:domains:config")

app = FastAPI()


def load_domains_from_redis() -> Optional[Dict[str, Any]]:
    if not redis:
        return None
    try:
        client = redis.from_url(REDIS_URL)
        data = client.get(REDIS_KEY)
        if not data:
            return None
        return json.loads(data)
    except Exception:
        return None


@app.get("/v1/domains")
async def list_domains() -> JSONResponse:
    # 1) Try Redis
    cfg = load_domains_from_redis()
    if cfg:
        # Expect cfg like {"domains": {"id": { ...config... }}}
        try:
            domains_map = cfg.get("domains") if isinstance(cfg, dict) else None
            if isinstance(domains_map, dict):
                data_list = []
                for did, conf in domains_map.items():
                    data_list.append({"id": did, "config": conf})
                return JSONResponse({"data": data_list})
        except Exception:
            pass
        # fallback raw
        return JSONResponse({"data": [], "config": cfg})

    # 2) Fallback: gather models as minimal info
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{LOCALAI_CORE}/v1/models")
            if r.status_code == 200:
                models = r.json()
                return JSONResponse({"data": [], "models": models})
        except Exception:
            pass
    return JSONResponse({"data": [], "models": {"data": []}}, status_code=200)


@app.get("/v1/metrics")
async def metrics() -> Response:
    # Proxy to prometheus metrics if available
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{LOCALAI_CORE}/metrics")
            return PlainTextResponse(r.text, status_code=r.status_code)
        except Exception:
            return PlainTextResponse("", status_code=204)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(path: str, request: Request) -> Response:
    # Generic proxy to LocalAI core
    url = f"{LOCALAI_CORE}/{path}".rstrip("/")
    method = request.method
    headers = dict(request.headers)
    body = await request.body()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.request(method, url, headers=headers, content=body)
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))
