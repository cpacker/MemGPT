from collections import UserDict
import json
import os
import inspect
from memgpt import MemGPT
from memgpt import constants
import memgpt.functions.function_sets.base as base_functions
from memgpt.functions.functions import USER_FUNCTIONS_DIR

from tests.utils import wipe_config

import pytest


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
        client = MemGPT(quickstart="openai")
    else:
        client = MemGPT(quickstart="memgpt_hosted")

    agent_state = client.create_agent(
        agent_config={
            # "name": test_agent_id,
            "persona": constants.DEFAULT_PERSONA,
            "human": constants.DEFAULT_HUMAN,
        }
    )

    return client.server._get_or_load_agent(user_id="NULL", agent_id=agent_state.id)


@pytest.fixture(scope="module")
def hello_world_function():
    with open(os.path.join(USER_FUNCTIONS_DIR, "hello_world.py"), "w") as f:
        f.write(inspect.getsource(hello_world))


@pytest.fixture(scope="module")
def ai_function_call():
    class AiFunctionCall(UserDict):
        def content(self):
            return self.data["content"]

    return AiFunctionCall(
        {
            "content": "I will now call hello world",
            "function_call": {
                "name": "hello_world",
                "arguments": json.dumps({}),
            },
        }
    )


def test_add_function_happy(agent, hello_world_function, ai_function_call):
    agent.add_function("hello_world")

    assert "hello_world" in [f_schema["name"] for f_schema in agent.functions]
    assert "hello_world" in agent.functions_python.keys()

    msgs, heartbeat_req, function_failed = agent._handle_ai_response(ai_function_call)
    content = json.loads(msgs[-1]["content"])
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
