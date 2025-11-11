from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routers import documents
from app.core.config import get_settings
from app.core.neo4j import get_neo4j_driver

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
    )

    app.include_router(documents.router)

    @app.on_event("startup")
    async def startup_event() -> None:
        get_neo4j_driver()
        logger.info("DMS service started")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        driver = get_neo4j_driver()
        await driver.close()
        logger.info("DMS service stopped")

    return app


app = create_app()
