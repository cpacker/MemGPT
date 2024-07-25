from typing import TYPE_CHECKING, Generator, Optional

from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.orm import sessionmaker

from memgpt.settings import BackendConfiguration, settings

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def create_engine(backend_configuration: Optional["BackendConfiguration"] = None) -> "Engine":
    """creates an engine for the storage_type designated by settings
    Args:
        backend_configuration: a BackendConfiguration object - this is a test hook, you should NOT be using this for application code!

    Returns: a sqlalchemy engine
    """
    backend = backend_configuration or settings.backend
    return sqlalchemy_create_engine(backend.database_uri)


def get_db_session() -> "Generator":
    """dependency primarily for FastAPI"""
    bound_session = sessionmaker(bind=create_engine())
    with bound_session() as session:
        yield session
