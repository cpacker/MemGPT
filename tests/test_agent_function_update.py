import inspect
import os

import pytest

from memgpt import create_client
from memgpt.functions.functions import USER_FUNCTIONS_DIR
from memgpt.schemas.message import Message
from memgpt.utils import assistant_function_to_tool, json_dumps, json_loads
from tests.mock_factory.models import MockTokenFactory, MockUserFactory


def hello_world(self) -> str:
    """Test function for agent to gain access to

    Returns:
        str: A message for the world
    """
    return "hello, world!"


@pytest.fixture(scope="module")
def agent(request, db_session, test_app):
    """Create a test agent that we can call functions on"""
    requesting_user = MockUserFactory(db_session=db_session).generate()
    api_token = MockTokenFactory(db_session=db_session, user_id=requesting_user.id).generate()
    token = api_token.api_key
    client_args = {"base_url": "http://test", "token": token, "debug": True, "app": test_app}

    # create memgpt client
    client = create_client(**client_args)

    agent_state = client.create_agent()

    return client.server._get_or_load_agent(user_id=requesting_user.id, agent_id=agent_state.id)


@pytest.fixture(scope="module")
def hello_world_function():
    with open(os.path.join(USER_FUNCTIONS_DIR, "hello_world.py"), "w", encoding="utf-8") as f:
        f.write(inspect.getsource(hello_world))


@pytest.fixture(scope="module")
def ai_function_call():
    return Message(
        **assistant_function_to_tool(
            {
                "role": "assistant",
                "content": "I will now call hello world",
                "function_call": {
                    "name": "hello_world",
                    "arguments": json_dumps({}),
                },
            }
        )
    )


def test_add_function_happy(agent, hello_world_function, ai_function_call):
    agent.add_function("hello_world")

    assert "hello_world" in [f_schema["name"] for f_schema in agent.functions]
    assert "hello_world" in agent.functions_python.keys()

    msgs, heartbeat_req, function_failed = agent._handle_ai_response(ai_function_call)
    content = json_loads(msgs[-1].to_openai_dict()["content"])
    assert content["message"] == "hello, world!"
    assert content["status"] == "OK"
    assert not function_failed


def test_add_function_already_loaded(agent, hello_world_function):
    agent.add_function("hello_world")
    # no exception for duplicate loading
    agent.add_function("hello_world")


def test_add_function_not_exist(agent):
    # pytest assert exception
    with pytest.raises(ValueError):
        agent.add_function("non_existent")


def test_remove_function_happy(agent, hello_world_function):
    agent.add_function("hello_world")

    # ensure function is loaded
    assert "hello_world" in [f_schema["name"] for f_schema in agent.functions]
    assert "hello_world" in agent.functions_python.keys()

    agent.remove_function("hello_world")

    assert "hello_world" not in [f_schema["name"] for f_schema in agent.functions]
    assert "hello_world" not in agent.functions_python.keys()


def test_remove_function_not_exist(agent):
    # do not raise error
    agent.remove_function("non_existent")


def test_remove_base_function_fails(agent):
    with pytest.raises(ValueError):
        agent.remove_function("send_message")


if __name__ == "__main__":
    pytest.main(["-vv", os.path.abspath(__file__)])
