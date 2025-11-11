from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterator

from sqlmodel import Session, SQLModel, create_engine

from ..config import get_settings

settings = get_settings()

connect_args: dict[str, object] = {}
if settings.is_sqlite:
    connect_args["check_same_thread"] = False
elif settings.is_postgres:
    connect_args = {}

engine = create_engine(
    str(settings.database_url),
    echo=settings.database_echo,
    connect_args=connect_args,
)


def init_db() -> None:
    """
    Create SQLModel tables on the configured engine.
    """
    # Import models on demand to ensure the SQLModel metadata is populated.
    from ..models import flow  # noqa: F401

    SQLModel.metadata.create_all(engine)


def SessionLocal() -> Session:
    """
    Convenience factory matching typical FastAPI dependency patterns.
    """
    return Session(engine)


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency injection helper.
    """
    with Session(engine) as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Provide a transactional scope for imperative code paths.
    """
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
