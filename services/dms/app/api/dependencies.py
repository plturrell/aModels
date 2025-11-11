from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.neo4j import get_neo4j_driver
from app.core.redis import get_redis_client


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session() as session:
        yield session


def get_redis():
    return get_redis_client()


def get_neo4j():
    return get_neo4j_driver()
