from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit
import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


from memgpt.settings import settings, BackendConfiguration
from memgpt.orm.utilities import create_engine
from memgpt.orm.__all__ import Base
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.server import SyncServer
from memgpt.server.rest_api.app import app

from tests.mock_factory.models import MockAgentFactory, MockUserFactory


if TYPE_CHECKING:
    from sqlalchemy import Session


# new ORM
@pytest.fixture(params=["sqlite_chroma","postgres",])
def db_session(request) -> "Session":
    """Creates a function-scoped orm session for the given test and adapter.
    Note: both pg and sqlite/chroma will have results scoped to each test function - so 2x results
    for each. These are cleared at the _beginning_ of each test run - so states are persisted for inspection
    after the end of the test.

    """
    function_ = request.node.name.replace("[","_").replace("]","_").replace("-","_").strip("_")
    adapter_test_configurations = {
        "sqlite_chroma": {
            "statements": (text(f"attach ':memory:' as {function_}"),),
            "database": f"/sqlite/{function_}.db"
        },
        "postgres": {
            "statements":(
                text(f"CREATE SCHEMA IF NOT EXISTS {function_}"),
                text(f"CREATE EXTENSION IF NOT EXISTS vector"),
                text(f"SET search_path TO {function_},public"),
            ),
            "database": "test_memgpt"
        }
    }
    adapter = adapter_test_configurations[request.param]
    # update the db uri to reflect the test function and param
    match request.param:
        case "sqlite_chroma":
            database_uri = f"sqlite:///{adapter['database']}"
        case "postgres":
            url_parts = list(urlsplit(settings.backend.database_uri))
            PATH_PARAM = 2
            url_parts[PATH_PARAM] = f"/{adapter['database']}"
            database_uri = urlunsplit(url_parts)
    backend = BackendConfiguration(name=request.param,
                                   database_uri=database_uri)
    engine = create_engine(backend)

    with engine.begin() as connection:
        for statement in adapter["statements"]:
            connection.execute(statement)
        Base.metadata.drop_all(bind=connection)
        Base.metadata.create_all(bind=connection)
    with sessionmaker(bind=engine)() as session:
        yield session

@pytest.fixture
def server(db_session):
    return SyncServer(db_session=db_session)

@pytest.fixture
def test_app(server):
    """a per-test-function db scoped version of the rest api app"""
    app.dependency_overrides[get_memgpt_server] = lambda : server
    return app

@pytest.fixture
def user_and_agent_seed(db_session):
    """ seed a single user and an Agent for that user
    """
    user = MockUserFactory(db_session=db_session).generate()
    agent = MockAgentFactory(db_session=db_session, user_id=user.id).generate()
    return user, agent
