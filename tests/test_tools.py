import os
import threading
import time
import uuid
from typing import Union

import pytest
from dotenv import load_dotenv

from letta import create_client
from letta.agent import Agent
from letta.client.client import LocalClient, RESTClient
from letta.constants import DEFAULT_PRESET
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory

test_agent_name = f"test_client_{str(uuid.uuid4())}"
# test_preset_name = "test_preset"
test_preset_name = DEFAULT_PRESET
test_agent_state = None
client = None

test_agent_state_post_message = None
test_user_id = uuid.uuid4()


def run_server():
    load_dotenv()

    # _reset_config()

    from letta.server.rest_api.app import start_server

    print("Starting server...")
    start_server(debug=True)


# Fixture to create clients with different configurations
@pytest.fixture(
    # params=[{"server": True}, {"server": False}],  # whether to use REST API server
    params=[{"server": True}],  # whether to use REST API server
    scope="module",
)
def client(request):

    if request.param["server"]:
        # get URL from enviornment
        server_url = os.getenv("MEMGPT_SERVER_URL")
        if server_url is None:
            # run server in thread
            server_url = "http://localhost:8283"
            print("Starting server thread")
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(5)
        print("Running client tests with server:", server_url)
    else:
        assert False, "Local client not implemented"

    assert server_url is not None
    client = create_client(base_url=server_url)  # This yields control back to the test function
    client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))
    # Clear all records from the Tool table
    yield client


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client):
    agent_state = client.create_agent(name=test_agent_name)
    print("AGENT ID", agent_state.id)
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


def test_create_tool(client: Union[LocalClient, RESTClient]):
    """Test creation of a simple tool"""

    def print_tool(message: str):
        """
        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.

        """
        print(message)
        return message

    tools = client.list_tools()
    assert sorted([t.name for t in tools]) == sorted(
        [
            "archival_memory_search",
            "send_message",
            "conversation_search",
            "conversation_search_date",
            "archival_memory_insert",
        ]
    )

    tool = client.create_tool(print_tool, name="my_name", tags=["extras"])

    tools = client.list_tools()
    assert tool in tools, f"Expected {tool.name} in {[t.name for t in tools]}"
    print(f"Updated tools {[t.name for t in tools]}")

    # check tool id
    tool = client.get_tool(tool.id)
    assert tool is not None, "Expected tool to be created"
    assert tool.id == tool.id, f"Expected {tool.id} to be {tool.id}"

    # create agent with tool
    agent_state = client.create_agent(tools=[tool.name])

    # Send message without error
    client.user_message(agent_id=agent_state.id, message="hi")


def test_create_agent_tool(client):
    """Test creation of a agent tool"""

    def core_memory_clear(self: "Agent"):
        """
        Args:
            agent (Agent): The agent to delete from memory.

        Returns:
            str: The agent that was deleted.

        """
        self.memory.update_block_value(label="human", value="")
        self.memory.update_block_value(label="persona", value="")
        print("UPDATED MEMORY", self.memory.memory)
        return None

    # TODO: test attaching and using function on agent
    tool = client.create_tool(core_memory_clear, tags=["extras"])
    print(f"Created tool", tool.name)

    # create agent with tool
    memory = ChatMemory(human="I am a human", persona="You must clear your memory if the human instructs you")
    agent = client.create_agent(name=test_agent_name, tools=[tool.name], memory=memory)
    assert str(tool.created_by_id) == str(agent.user_id), f"Expected {tool.created_by_id} to be {agent.user_id}"

    # initial memory
    initial_memory = client.get_in_context_memory(agent.id)
    print("initial memory", initial_memory.compile())
    human = initial_memory.get_block("human")
    persona = initial_memory.get_block("persona")
    print("Initial memory:", human, persona)
    assert len(human) > 0, "Expected human memory to be non-empty"
    assert len(persona) > 0, "Expected persona memory to be non-empty"

    # test agent tool
    response = client.send_message(role="user", agent_id=agent.id, message="clear your memory with the core_memory_clear tool")
    print(response)

    # updated memory
    print("Query agent memory")
    updated_memory = client.get_in_context_memory(agent.id)
    human = updated_memory.get_block("human")
    persona = updated_memory.get_block("persona")
    print("Updated memory:", human, persona)
    assert len(human) == 0, "Expected human memory to be empty"
    assert len(persona) == 0, "Expected persona memory to be empty"


def test_custom_import_tool(client):
    pass
