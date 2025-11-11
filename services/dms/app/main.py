from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from app.api.routers import documents, health
from app.core.auth import AuthMiddleware
from app.core.config import get_settings
from app.core.middleware import CorrelationIDMiddleware
from app.core.neo4j import get_neo4j_driver

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
    )

    # Add correlation ID middleware
    app.add_middleware(CorrelationIDMiddleware)
    
    # Add authentication middleware if enabled
    require_auth = os.getenv("DMS_REQUIRE_AUTH", "false").lower() == "true"
    if require_auth:
        logger.info("Authentication enabled - all requests require valid credentials")
        app.add_middleware(AuthMiddleware, require_auth=True)
    else:
        logger.info("Authentication optional - running in development mode")
        app.add_middleware(AuthMiddleware, require_auth=False)
    
    # Register routers
    app.include_router(health.router)
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
