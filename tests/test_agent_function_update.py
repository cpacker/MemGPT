import os

import pytest

from letta import create_client
from letta.schemas.message import Message
from letta.utils import assistant_function_to_tool, json_dumps
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
        create_config("letta_hosted")

    # create letta client
    client = create_client()

    agent_state = client.create_agent()

    return client.server._get_or_load_agent(agent_id=agent_state.id)


@pytest.fixture(scope="module")
def ai_function_call():
    return Message(
        **assistant_function_to_tool(
            {
                "role": "assistant",
                "text": "I will now call hello world",  # TODO: change to `content` once `Message` is updated
                "function_call": {
                    "name": "hello_world",
                    "arguments": json_dumps({}),
                },
            }
        )
    )
