from typing import Optional, TYPE_CHECKING, Generator
from urllib.parse import urlsplit, urlunsplit
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.orm import sessionmaker


from memgpt.settings import settings, BackendConfiguration

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def create_engine(
        backend_configuration: Optional["BackendConfiguration"] = None
) -> "Engine":
    """creates an engine for the storage_type designated by settings
    Args:
        backend_configuration: a BackendConfiguration object - this is a test hook, you should NOT be using this for application code!

    Returns: a sqlalchemy engine
    """
    backend = backend_configuration or settings.backend
    return sqlalchemy_create_engine(backend.database_uri)



def get_db_session() -> "Generator":
    """The raw session"""
    return sessionmaker(bind=create_engine())

def get_db_session_generator() -> "Generator":
    """dependency primarily for FastAPI"""
    bound_session = get_db_session()
    with bound_session() as session:
        yield session