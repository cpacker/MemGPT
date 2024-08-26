import os
import threading
import time
import uuid

import pytest
from dotenv import load_dotenv

from memgpt import Admin, create_client
from memgpt.agent import Agent
from memgpt.constants import DEFAULT_PRESET
from memgpt.schemas.memory import ChatMemory

test_agent_name = f"test_client_{str(uuid.uuid4())}"
# test_preset_name = "test_preset"
test_preset_name = DEFAULT_PRESET
test_agent_state = None
client = None

test_agent_state_post_message = None
test_user_id = uuid.uuid4()


# admin credentials
test_server_token = "test_server_token"


def run_server():
    load_dotenv()

    # _reset_config()

    from memgpt.server.rest_api.server import start_server

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
            # NOTE: must set MEMGPT_SERVER_PASS enviornment variable
            server_url = "http://localhost:8283"
            print("Starting server thread")
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(5)
        print("Running client tests with server:", server_url)
        # create user via admin client
        admin = Admin(server_url, test_server_token)
        user = admin.create_user()  # Adjust as per your client's method
        api_key = admin.create_key(user.id)
    else:
        # use local client (no server)
        assert False, "Local client not implemented"
        server_url = None

    assert server_url is not None
    assert api_key.key is not None
    client = create_client(base_url=server_url, token=api_key.key)  # This yields control back to the test function
    try:
        yield client
    finally:
        # cleanup user
        if server_url:
            admin.delete_user(user.id)


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client):
    agent_state = client.create_agent(name=test_agent_name)
    print("AGENT ID", agent_state.id)
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


def test_create_tool(client):
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
    print(f"Original tools {[t.name for t in tools]}")

    tool = client.create_tool(print_tool, tags=["extras"])

    tools = client.list_tools()
    assert tool in tools, f"Expected {tool.name} in {[t.name for t in tools]}"
    print(f"Updated tools {[t.name for t in tools]}")

    # check tool id
    tool = client.get_tool(tool.name)


# TODO: add back once we fix admin client tool creation
# def test_create_agent_tool_admin(admin_client):
#    if admin_client is None:
#        return
#
#    def print_tool(message: str):
#        """
#        Args:
#            message (str): The message to print.
#
#        Returns:
#            str: The message that was printed.
#
#        """
#        print(message)
#        return message
#
#    tools = admin_client.list_tools()
#    print(f"Original tools {[t.name for t in tools]}")
#
#    tool = admin_client.create_tool(print_tool, tags=["extras"])
#
#    tools = admin_client.list_tools()
#    assert tool in tools, f"Expected {tool.name} in {[t.name for t in tools]}"
#    print(f"Updated tools {[t.name for t in tools]}")
#
#    # check tool id
#    tool = admin_client.get_tool(tool.name)
#    assert tool.user_id is None, f"Expected {tool.user_id} to be None"


def test_create_agent_tool(client):
    """Test creation of a agent tool"""

    def core_memory_clear(self: Agent):
        """
        Args:
            agent (Agent): The agent to delete from memory.

        Returns:
            str: The agent that was deleted.

        """
        self.memory.update_block_value(name="human", value="")
        self.memory.update_block_value(name="persona", value="")
        print("UPDATED MEMORY", self.memory.memory)
        return None

    # TODO: test attaching and using function on agent
    tool = client.create_tool(core_memory_clear, tags=["extras"], update=True)
    print(f"Created tool", tool.name)

    # create agent with tool
    memory = ChatMemory(human="I am a human", persona="You must clear your memory if the human instructs you")
    agent = client.create_agent(name=test_agent_name, tools=[tool.name], memory=memory)
    assert str(tool.user_id) == str(agent.user_id), f"Expected {tool.user_id} to be {agent.user_id}"

    # initial memory
    initial_memory = client.get_in_context_memory(agent.id)
    print("initial memory", initial_memory)
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
