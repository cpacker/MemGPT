import inspect
import json
import os
import uuid

import pytest

from memgpt import constants, create_client
from memgpt.functions.functions import USER_FUNCTIONS_DIR
from memgpt.models import chat_completion_response
from memgpt.utils import assistant_function_to_tool
from tests import TEST_MEMGPT_CONFIG
from tests.utils import create_config, wipe_config


def hello_world(self) -> str:
    """Test function for agent to gain access to

    Returns:
        str: A message for the world
    """
    return "hello, world!"


@pytest.fixture(scope="module")
def agent():
    """Create a test agent that we can call functions on"""
    wipe_config()
    global client
    if os.getenv("OPENAI_API_KEY"):
        create_config("openai")
    else:
        create_config("memgpt_hosted")

    # create memgpt client
    client = create_client()

    # ensure user exists
    user_id = uuid.UUID(TEST_MEMGPT_CONFIG.anon_clientid)
    if not client.server.get_user(user_id=user_id):
        client.server.create_user({"id": user_id})

    agent_state = client.create_agent()

    return client.server._get_or_load_agent(user_id=user_id, agent_id=agent_state.id)


@pytest.fixture(scope="module")
def hello_world_function():
    with open(os.path.join(USER_FUNCTIONS_DIR, "hello_world.py"), "w", encoding="utf-8") as f:
        f.write(inspect.getsource(hello_world))


@pytest.fixture(scope="module")
def ai_function_call():
    return chat_completion_response.Message(
        **assistant_function_to_tool(
            {
                "role": "assistant",
                "content": "I will now call hello world",
                "function_call": {
                    "name": "hello_world",
                    "arguments": json.dumps({}),
                },
            }
        )
    )

    return


def test_add_function_happy(agent, hello_world_function, ai_function_call):
    agent.add_function("hello_world")

    assert "hello_world" in [f_schema["name"] for f_schema in agent.functions]
    assert "hello_world" in agent.functions_python.keys()

    msgs, heartbeat_req, function_failed = agent._handle_ai_response(ai_function_call)
    content = json.loads(msgs[-1].to_openai_dict()["content"], strict=constants.JSON_LOADS_STRICT)
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
