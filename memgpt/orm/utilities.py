from typing import Optional, TYPE_CHECKING, Generator
from urllib.parse import urlsplit, urlunsplit
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.orm import sessionmaker


from memgpt.settings import settings

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def create_engine(
        storage_type: Optional[str] = None,
        database: Optional[str] = None,
) -> "Engine":
    """creates an engine for the storage_type designated by settings
    Args:
        storage_type: test hook to inject storage_type, you should not be setting this
        database: test hook to inject database, you should not be setting this
    Returns: a sqlalchemy engine
    """
    storage_type = storage_type or settings.storage_type
    match storage_type:
        case "postgres":
            url_parts = list(urlsplit(settings.pg_uri))
            PATH_PARAM = 2 # avoid the magic number!
            url_parts[PATH_PARAM] = f"/{database}" if database else url_parts.path
            return sqlalchemy_create_engine(urlunsplit(url_parts))
        case "sqlite-chroma":
            return sqlalchemy_create_engine(f"sqlite:///{database}")
        case _:
            raise ValueError(f"Unsupported storage_type: {storage_type}")


def get_db_session() -> "Generator":
    """dependency primarily for FastAPI"""
    bound_session = sessionmaker(bind=create_engine())
    with bound_session() as session:
        yield session