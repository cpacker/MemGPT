from typing import TYPE_CHECKING, Callable
from urllib.parse import urlsplit, urlunsplit
import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from memgpt import create_client
from memgpt.settings import settings, BackendConfiguration
from memgpt.orm.utilities import create_engine
from memgpt.orm.__all__ import Base, Tool, User
from memgpt.orm.utilities import get_db_session
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.server import SyncServer
from memgpt.server.rest_api.app import app

from tests.mock_factory.models import (
    MockUserFactory,
    MockOrganizationFactory,
    MockTokenFactory,
    MockAgentFactory,
)


if TYPE_CHECKING:
    from sqlalchemy import Session


# new ORM
@pytest.fixture(params=["postgres",])
def test_session_maker(request):
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
                text(f"DROP SCHEMA IF EXISTS {function_} CASCADE"),
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

    def _session_maker():
        return sessionmaker(bind=engine)

    return _session_maker

@pytest.fixture
def test_get_db_session(test_session_maker) -> Callable:
    return test_session_maker()

@pytest.fixture
def db_session(test_get_db_session) -> "Session":
    with test_get_db_session() as session:
        yield session

@pytest.fixture
def server(db_session):
    return SyncServer(db_session=db_session)

@pytest.fixture
def test_app(server, db_session):
    """a per-test-function db scoped version of the rest api app"""
    app.dependency_overrides[get_memgpt_server] = lambda : server
    app.dependency_overrides[get_db_session] = lambda : db_session
    Tool.load_default_tools(db_session)
    return app

@pytest.fixture
def user_and_agent_seed(db_session):
    """ seed a single user and an Agent for that user
    """
    user = MockUserFactory(db_session=db_session).generate()
    agent = MockAgentFactory(db_session=db_session, user_id=user.id).generate()
    return user, agent

# Fixture to create clients with different configurations
@pytest.fixture(
    params=[{"server": True}],  # whether to use REST API server
)
def client(request, db_session, test_app):
    if request.param["server"]:

        api_token = MockTokenFactory(db_session=db_session, user_id=User.default(db_session).id).generate()
        token = api_token.api_key
        client_args = {
            "base_url": "http://test",
            "token": token,
            "debug": True,
            "app": test_app
        }
    else:
        # use local client (no server)
        client_args = {
            "token": None,
            "base_url": None
        }
    yield create_client(**client_args)


@pytest.fixture(autouse=True)
def patch_local_db_calls(test_get_db_session: Callable):
    # TODO: I'm not happy about this, but it's either make the tests messy without hooks,
    # or make the code messy with hooks. I'm choosing the former for now.
    with (
        patch(
            "memgpt.metadata.get_db_session",
            test_get_db_session,
        ),
        patch(
            "memgpt.agent_store.storage.get_db_session",
            test_get_db_session,
        ),
        patch(
            "memgpt.server.server.get_db_session",
            test_get_db_session,
        ),
        patch(
            "memgpt.server.rest_api.app.get_db_session",
            test_get_db_session,
        ),
        patch(
            "memgpt.server.rest_api.utils.get_db_session",
            test_get_db_session,
        ),
    ):
        yield