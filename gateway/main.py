import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))

HANA_URL = os.getenv("HANA_URL", "http://localhost:8083")

client = httpx.AsyncClient(timeout=30.0)

app = FastAPI(title="aModels Gateway", version="0.1.0")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)


