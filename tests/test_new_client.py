import pytest

from memgpt import create_client
from memgpt.schemas.memory import ChatMemory


@pytest.fixture(scope="module")
def client():
    yield create_client()


@pytest.fixture(scope="module")
def agent(client):
    agent_state = client.create_agent(name="test_agent")
    yield agent_state

    client.delete_agent(agent_state.id)
    assert client.get_agent(agent_state.id) is None, f"Failed to properly delete agent {agent_state.id}"


def test_agent(client):

    # create agent
    agent_state_test = client.create_agent(
        name="test_agent",
        memory=ChatMemory(human="I am a human", persona="I am an agent"),
        description="This is a test agent",
    )

    # list agents
    agents = client.list_agents()
    assert agent_state_test.id in [a.id for a in agents]

    # get agent
    agent_state = client.get_agent(agent_state_test.id)
    assert agent_state.name == "test_agent"

    # update agent: name
    new_name = "new_agent"
    client.update_agent(agent_state_test.id, name=new_name)
    assert client.get_agent(agent_state_test.id).name == new_name

    # update agent: system prompt
    new_system_prompt = agent_state.system + "Always respond with a !"
    client.update_agent(agent_state_test.id, system=new_system_prompt)
    assert client.get_agent(agent_state_test.id).system == new_system_prompt

    # update agent: tools
    # update agent: memory
    # update agent: message_ids
    # update agent: llm config
    # update agent: embedding config

    # delete agent


def test_tools(client):

    def print_tool(message: str):
        """
        A tool to print a message

        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.

        """
        print(message)
        return message

    def print_tool2(msg: str):
        """
        Another tool to print a message

        Args:
            msg (str): The message to print.
        """
        print(msg)

    # create tool
    orig_tool_length = len(client.list_tools())
    tool = client.create_tool(print_tool, tags=["extras"])

    # list tools
    tools = client.list_tools()
    assert tool.name in [t.name for t in tools]

    # get tool id
    assert tool.id == client.get_tool_id(name="print_tool")

    # update tool: extras
    extras2 = ["extras2"]
    client.update_tool(tool.id, tags=extras2)
    assert client.get_tool(tool.id).tags == extras2

    # update tool: source code
    client.update_tool(tool.id, name="print_tool2", func=print_tool2)
    assert client.get_tool(tool.id).name == "print_tool2"

    # delete tool
    client.delete_tool(tool.id)
    assert len(client.list_tools()) == orig_tool_length
