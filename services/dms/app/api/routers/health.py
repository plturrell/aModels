"""Health check endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_factory
from app.core.neo4j import get_neo4j_driver
from app.core.redis import get_redis_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


async def check_postgres() -> tuple[bool, str]:
    """Check PostgreSQL connectivity."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            return True, "ok"
    except Exception as e:
        logger.error("PostgreSQL health check failed: %s", e)
        return False, str(e)


async def check_redis() -> tuple[bool, str]:
    """Check Redis connectivity."""
    try:
        client = get_redis_client()
        await client.ping()
        return True, "ok"
    except Exception as e:
        logger.error("Redis health check failed: %s", e)
        return False, str(e)


async def check_neo4j() -> tuple[bool, str]:
    """Check Neo4j connectivity."""
    try:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS num")
            await result.single()
            return True, "ok"
    except Exception as e:
        logger.error("Neo4j health check failed: %s", e)
        return False, str(e)


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint.
    
    Returns 200 if service is running, regardless of dependency status.
    Use /healthz/detailed for dependency checks.
    """
    return {
        "status": "healthy",
        "service": "dms"
    }


@router.get("/healthz/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check() -> dict[str, Any]:
    """Detailed health check with all dependencies.
    
    Returns 200 if service and all dependencies are healthy.
    Returns 503 if any dependency is unhealthy.
    """
    # Run all checks in parallel
    results = await asyncio.gather(
        check_postgres(),
        check_redis(),
        check_neo4j(),
        return_exceptions=True
    )
    
    postgres_ok, postgres_msg = results[0] if not isinstance(results[0], Exception) else (False, str(results[0]))
    redis_ok, redis_msg = results[1] if not isinstance(results[1], Exception) else (False, str(results[1]))
    neo4j_ok, neo4j_msg = results[2] if not isinstance(results[2], Exception) else (False, str(results[2]))
    
    all_healthy = postgres_ok and redis_ok and neo4j_ok
    
    response = {
        "status": "healthy" if all_healthy else "degraded",
        "service": "dms",
        "dependencies": {
            "postgres": {
                "status": "up" if postgres_ok else "down",
                "message": postgres_msg
            },
            "redis": {
                "status": "up" if redis_ok else "down",
                "message": redis_msg
            },
            "neo4j": {
                "status": "up" if neo4j_ok else "down",
                "message": neo4j_msg
            }
        }
    }
    
    # Return 503 if any dependency is down
    if not all_healthy:
        from fastapi import Response
        return Response(
            content=str(response),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json"
        )
    
    return response


@router.get("/readyz", status_code=status.HTTP_200_OK)
async def readiness_check() -> dict[str, Any]:
    """Readiness check for Kubernetes.
    
    Returns 200 if service is ready to accept traffic.
    Returns 503 if service is not ready.
    """
    # Check critical dependencies only (postgres for now)
    postgres_ok, postgres_msg = await check_postgres()
    
    if not postgres_ok:
        from fastapi import Response
        return Response(
            content='{"status": "not_ready", "reason": "database unavailable"}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json"
        )
    
    return {
        "status": "ready",
        "service": "dms"
    }


@router.get("/livez", status_code=status.HTTP_200_OK)
async def liveness_check() -> dict[str, Any]:
    """Liveness check for Kubernetes.
    
    Returns 200 if service process is alive.
    """
    return {
        "status": "alive",
        "service": "dms"
    }
