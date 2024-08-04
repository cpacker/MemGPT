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

    tools = client.list_tools()

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
    print("TOOLS", [t.name for t in tools])
    agent_state = client.get_agent(agent_state_test.id)
    assert agent_state.name == "test_agent"

    # update agent: name
    new_name = "new_agent"
    client.update_agent(agent_state_test.id, name=new_name)
    assert client.get_agent(agent_state_test.id).name == new_name

    ## update agent: system prompt
    # new_system_prompt = agent_state.system + "Always respond with a !"
    # client.update_agent(agent_state_test.id, system=new_system_prompt)
    # assert client.get_agent(agent_state_test.id).system == new_system_prompt

    # update agent: tools
    # update agent: memory
    # update agent: message_ids
    # update agent: llm config
    # update agent: embedding config

    # delete agent
    client.delete_agent(agent_state_test.id)


def test_memory(client, agent):

    # get agent memory
    original_memory = client.get_in_context_memory(agent.id)

    # update core memory
    updated_memory = client.update_in_context_memory(agent.id, section="human", content="I am a human")

    # get memory
    assert updated_memory.memory["human"].value != original_memory.memory["human"].value  # check if the memory has been updated
    assert updated_memory.id == original_memory.id  # memory id should remain the same


def test_archival_memory(client, agent):
    """Test functions for interacting with archival memory store"""

    # add archival memory
    memory_str = "I love chats"
    passage = client.insert_archival_memory(agent.id, memory=memory_str)

    # list archival memory
    passages = client.list_archival_memory(agent.id)
    assert passage.message_str in [p.message_str for p in passages], f"Missing passage {passage.message_str} in {passages}"

    # delete archival memory
    client.delete_archival_memory(agent.id, passage.id)


def test_recall_memory(client, agent):
    """Test functions for interacting with recall memory store"""

    # send message to the agent
    message_str = "Hello"
    message = client.send_message(agent.id, message_str)

    # list messages
    messages = client.list_messages(agent.id)
    assert message.message_str in [m.message_str for m in messages], f"Missing message {message.message_str} in {messages}"

    # get in-context messages
    in_context_messages = client.get_in_context_messages(agent.id)
    assert message.id in [m.id for m in in_context_messages], f"Missing message {message.id} in {in_context_messages}"


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


def test_sources(client, agent):

    # list sources (empty)
    sources = client.list_sources()
    assert len(sources) == 0

    # create a source
    test_source_name = "test_source"
    source = client.create_source(name=test_source_name)

    # list sources
    sources = client.list_sources()
    assert len(sources) == 1
    assert sources[0].metadata_["num_passages"] == 0
    assert sources[0].metadata_["num_documents"] == 0

    # update the source
    original_id = source.id
    original_name = source.name
    new_name = original_name + "_new"
    client.update_source(source_id=source.id, name=new_name)

    # get the source name (check that it's been updated)
    source = client.get_source(source_id=source.id)
    assert source.name == new_name
    assert source.id == original_id

    # get the source id (make sure that it's the same)
    assert str(original_id) == client.get_source_id(source_name=new_name)

    # TODO reenable once archival works

    # # check agent archival memory size
    # archival_memories = client.get_agent_archival_memory(agent_id=agent.id).archival_memory
    # print(archival_memories)
    # assert len(archival_memories) == 0

    # # load a file into a source
    # filename = "CONTRIBUTING.md"
    # upload_job = client.load_file_into_source(filename=filename, source_id=source.id)
    # print("Upload job", upload_job, upload_job.status, upload_job.metadata)

    # # TODO: make sure things run in the right order
    # archival_memories = client.get_agent_archival_memory(agent_id=agent.id).archival_memory
    # assert len(archival_memories) == 0

    # attach a source
    client.attach_source_to_agent(source_id=source.id, agent_id=agent.id)

    # # list archival memory
    # archival_memories = client.get_agent_archival_memory(agent_id=agent.id).archival_memory
    # # print(archival_memories)
    # assert len(archival_memories) == 20 or len(archival_memories) == 21

    # check number of passages
    # sources = client.list_sources()
    # assert sources.sources[0].metadata_["num_passages"] > 0
    # assert sources.sources[0].metadata_["num_documents"] == 0  # TODO: fix this once document store added
    # print(sources)

    # detach the source
    # TODO: add when implemented
    # client.detach_source(source.name, agent.id)

    # delete the source
    client.delete_source(source.id)