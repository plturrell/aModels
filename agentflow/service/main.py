from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .cache import get_redis_client
from .config import get_settings
from .db import init_db
from .db.hana import ensure_hana_registry
from .db.postgres import ensure_postgres_registry
from .routers import flows_router, health_router, sgmi_router
from .services import FlowCatalog
from .services.langflow import LangflowClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    hana_ready = ensure_hana_registry()
    postgres_ready = ensure_postgres_registry()
    if hana_ready:
        logger.info("HANA registry table verified.")
    else:
        logger.warning("HANA registry table skipped (disabled or driver missing).")
    if postgres_ready:
        logger.info("PostgreSQL registry table verified.")
    else:
        logger.warning(
            "PostgreSQL registry table skipped (disabled or driver missing)."
        )

    catalog = FlowCatalog(settings.flows_dir)
    redis_client = get_redis_client()
    langflow_client = LangflowClient(
        base_url=settings.langflow_base_url,
        api_key=settings.langflow_api_key,
        auth_token=settings.langflow_auth_token,
        timeout_seconds=settings.langflow_timeout_seconds,
    )

    app.state.settings = settings
    app.state.catalog = catalog
    app.state.redis = redis_client
    app.state.langflow_client = langflow_client

    try:
        yield
    finally:
        await langflow_client.close()
        if redis_client is not None:
            close_callable = getattr(redis_client, "close", None)
            if callable(close_callable):
                result = close_callable()
                if result is not None:
                    await result


app = FastAPI(title="AgentFlow Service", version="0.1.0", lifespan=lifespan)

settings = get_settings()
allowed_origins = [origin.strip() for origin in settings.allow_origins.split(",") if origin.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health_router)
app.include_router(flows_router)
app.include_router(sgmi_router)
