import uuid

import pytest

from letta import create_client
from letta.client.client import LocalClient
from letta.schemas.agent import AgentState
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import BasicBlockMemory, ChatMemory, Memory
from letta.schemas.tool import ToolCreate


@pytest.fixture(scope="module")
def client():
    client = create_client()
    client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))

    yield client


@pytest.fixture(scope="module")
def agent(client):
    # Generate uuid for agent name for this example
    namespace = uuid.NAMESPACE_DNS
    agent_uuid = str(uuid.uuid5(namespace, "test_new_client_test_agent"))

    agent_state = client.create_agent(name=agent_uuid)
    yield agent_state

    client.delete_agent(agent_state.id)
    assert client.get_agent(agent_state.id) is None, f"Failed to properly delete agent {agent_state.id}"


def test_agent(client: LocalClient):
    # create agent
    agent_state_test = client.create_agent(
        name="test_agent2",
        memory=ChatMemory(human="I am a human", persona="I am an agent"),
        description="This is a test agent",
    )
    assert isinstance(agent_state_test.memory, Memory)

    # list agents
    agents = client.list_agents()
    assert agent_state_test.id in [a.id for a in agents]

    # get agent
    tools = client.list_tools()
    print("TOOLS", [t.name for t in tools])
    agent_state = client.get_agent(agent_state_test.id)
    assert agent_state.name == "test_agent2"
    for block in agent_state.memory.to_dict()["memory"].values():
        db_block = client.server.block_manager.get_block_by_id(block.get("id"), actor=client.user)
        assert db_block is not None, "memory block not persisted on agent create"
        assert db_block.value == block.get("value"), "persisted block data does not match in-memory data"

    assert isinstance(agent_state.memory, Memory)
    # update agent: name
    new_name = "new_agent"
    client.update_agent(agent_state_test.id, name=new_name)
    assert client.get_agent(agent_state_test.id).name == new_name

    assert isinstance(agent_state.memory, Memory)
    # update agent: system prompt
    new_system_prompt = agent_state.system + "\nAlways respond with a !"
    client.update_agent(agent_state_test.id, system=new_system_prompt)
    assert client.get_agent(agent_state_test.id).system == new_system_prompt

    assert isinstance(agent_state.memory, Memory)
    # update agent: message_ids
    old_message_ids = agent_state.message_ids
    new_message_ids = old_message_ids.copy()[:-1]  # pop one
    assert len(old_message_ids) != len(new_message_ids)
    client.update_agent(agent_state_test.id, message_ids=new_message_ids)
    assert client.get_agent(agent_state_test.id).message_ids == new_message_ids

    assert isinstance(agent_state.memory, Memory)
    # update agent: tools
    tool_to_delete = "send_message"
    assert tool_to_delete in agent_state.tools
    new_agent_tools = [t_name for t_name in agent_state.tools if t_name != tool_to_delete]
    client.update_agent(agent_state_test.id, tools=new_agent_tools)
    assert client.get_agent(agent_state_test.id).tools == new_agent_tools

    assert isinstance(agent_state.memory, Memory)
    # update agent: memory
    new_human = "My name is Mr Test, 100 percent human."
    new_persona = "I am an all-knowing AI."
    new_memory = ChatMemory(human=new_human, persona=new_persona)
    assert agent_state.memory.get_block("human").value != new_human
    assert agent_state.memory.get_block("persona").value != new_persona

    client.update_agent(agent_state_test.id, memory=new_memory)
    assert client.get_agent(agent_state_test.id).memory.get_block("human").value == new_human
    assert client.get_agent(agent_state_test.id).memory.get_block("persona").value == new_persona

    # update agent: llm config
    new_llm_config = agent_state.llm_config.model_copy(deep=True)
    new_llm_config.model = "fake_new_model"
    new_llm_config.context_window = 1e6
    assert agent_state.llm_config != new_llm_config
    client.update_agent(agent_state_test.id, llm_config=new_llm_config)
    assert client.get_agent(agent_state_test.id).llm_config == new_llm_config
    assert client.get_agent(agent_state_test.id).llm_config.model == "fake_new_model"
    assert client.get_agent(agent_state_test.id).llm_config.context_window == 1e6

    # update agent: embedding config
    new_embed_config = agent_state.embedding_config.model_copy(deep=True)
    new_embed_config.embedding_model = "fake_embed_model"
    assert agent_state.embedding_config != new_embed_config
    client.update_agent(agent_state_test.id, embedding_config=new_embed_config)
    assert client.get_agent(agent_state_test.id).embedding_config == new_embed_config
    assert client.get_agent(agent_state_test.id).embedding_config.embedding_model == "fake_embed_model"

    # delete agent
    client.delete_agent(agent_state_test.id)


def test_agent_add_remove_tools(client: LocalClient, agent):
    # Create and add two tools to the client
    # tool 1
    from composio_langchain import Action

    github_tool = client.load_composio_tool(action=Action.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER)

    # assert both got added
    tools = client.list_tools()
    assert github_tool.id in [t.id for t in tools]

    # Assert that all combinations of tool_names, organization id are unique
    combinations = [(t.name, t.organization_id) for t in tools]
    assert len(combinations) == len(set(combinations))

    # create agent
    agent_state = agent
    curr_num_tools = len(agent_state.tools)

    # add both tools to agent in steps
    agent_state = client.add_tool_to_agent(agent_id=agent_state.id, tool_id=github_tool.id)

    # confirm that both tools are in the agent state
    # we could access it like agent_state.tools, but will use the client function instead
    # this is obviously redundant as it requires retrieving the agent again
    # but allows us to test the `get_tools_from_agent` pathway as well
    curr_tools = client.get_tools_from_agent(agent_state.id)
    curr_tool_names = [t.name for t in curr_tools]
    assert len(curr_tool_names) == curr_num_tools + 1
    assert github_tool.name in curr_tool_names

    # remove only the github tool
    agent_state = client.remove_tool_from_agent(agent_id=agent_state.id, tool_id=github_tool.id)

    # confirm that only one tool left
    curr_tools = client.get_tools_from_agent(agent_state.id)
    curr_tool_names = [t.name for t in curr_tools]
    assert len(curr_tool_names) == curr_num_tools
    assert github_tool.name not in curr_tool_names


def test_agent_with_shared_blocks(client: LocalClient):
    persona_block = client.create_block(template_name="persona", value="Here to test things!", label="persona")
    human_block = client.create_block(template_name="human", value="Me Human, I swear. Beep boop.", label="human")
    existing_non_template_blocks = [persona_block, human_block]

    existing_non_template_blocks_no_values = []
    for block in existing_non_template_blocks:
        block_copy = block.copy()
        block_copy.value = ""
        existing_non_template_blocks_no_values.append(block_copy)

    # create agent
    first_agent_state_test = None
    second_agent_state_test = None
    try:
        first_agent_state_test = client.create_agent(
            name="first_test_agent_shared_memory_blocks",
            memory=BasicBlockMemory(blocks=existing_non_template_blocks),
            description="This is a test agent using shared memory blocks",
        )
        assert isinstance(first_agent_state_test.memory, Memory)

        first_blocks_dict = first_agent_state_test.memory.to_dict()["memory"]
        assert persona_block.id == first_blocks_dict.get("persona", {}).get("id")
        assert human_block.id == first_blocks_dict.get("human", {}).get("id")
        client.update_in_context_memory(first_agent_state_test.id, section="human", value="I'm an analyst therapist.")

        # when this agent is created with the shared block references this agent's in-memory blocks should
        # have this latest value set by the other agent.
        second_agent_state_test = client.create_agent(
            name="second_test_agent_shared_memory_blocks",
            memory=BasicBlockMemory(blocks=existing_non_template_blocks_no_values),
            description="This is a test agent using shared memory blocks",
        )

        assert isinstance(second_agent_state_test.memory, Memory)
        second_blocks_dict = second_agent_state_test.memory.to_dict()["memory"]
        assert persona_block.id == second_blocks_dict.get("persona", {}).get("id")
        assert human_block.id == second_blocks_dict.get("human", {}).get("id")
        assert second_blocks_dict.get("human", {}).get("value") == "I'm an analyst therapist."

    finally:
        if first_agent_state_test:
            client.delete_agent(first_agent_state_test.id)
        if second_agent_state_test:
            client.delete_agent(second_agent_state_test.id)


def test_memory(client: LocalClient, agent: AgentState):
    # get agent memory
    original_memory = client.get_in_context_memory(agent.id)
    assert original_memory is not None
    original_memory_value = str(original_memory.get_block("human").value)

    # update core memory
    updated_memory = client.update_in_context_memory(agent.id, section="human", value="I am a human")

    # get memory
    assert updated_memory.get_block("human").value != original_memory_value  # check if the memory has been updated


def test_archival_memory(client: LocalClient, agent: AgentState):
    """Test functions for interacting with archival memory store"""

    # add archival memory
    memory_str = "I love chats"
    passage = client.insert_archival_memory(agent.id, memory=memory_str)[0]

    # list archival memory
    passages = client.get_archival_memory(agent.id)
    assert passage.text in [p.text for p in passages], f"Missing passage {passage.text} in {passages}"

    # delete archival memory
    client.delete_archival_memory(agent.id, passage.id)


def test_recall_memory(client: LocalClient, agent: AgentState):
    """Test functions for interacting with recall memory store"""

    # send message to the agent
    message_str = "Hello"
    client.send_message(message=message_str, role="user", agent_id=agent.id)

    # list messages
    messages = client.get_messages(agent.id)
    exists = False
    for m in messages:
        if message_str in str(m):
            exists = True
    assert exists

    # get in-context messages
    messages = client.get_in_context_messages(agent.id)
    exists = False
    for m in messages:
        if message_str in str(m):
            exists = True
    assert exists


def test_tools(client: LocalClient):
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
    tool = client.create_tool(func=print_tool, tags=["extras"])

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


def test_tools_from_composio_basic(client: LocalClient):
    from composio_langchain import Action

    # Create a `LocalClient` (you can also use a `RESTClient`, see the letta_rest_client.py example)
    client = create_client()

    # create tool
    tool = client.load_composio_tool(action=Action.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER)

    # list tools
    tools = client.list_tools()
    assert tool.name in [t.name for t in tools]

    # We end the test here as composio requires login to use the tools
    # The tool creation includes a compile safety check, so if this test doesn't error out, at least the code is compilable


def test_tools_from_langchain(client: LocalClient):
    # create langchain tool
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    langchain_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Add the tool
    tool = client.load_langchain_tool(
        langchain_tool, additional_imports_module_attr_map={"langchain_community.utilities": "WikipediaAPIWrapper"}
    )

    # list tools
    tools = client.list_tools()
    assert tool.name in [t.name for t in tools]

    # get tool
    tool_id = client.get_tool_id(name=tool.name)
    retrieved_tool = client.get_tool(tool_id)
    source_code = retrieved_tool.source_code

    # Parse the function and attempt to use it
    local_scope = {}
    exec(source_code, {}, local_scope)
    func = local_scope[tool.name]

    expected_content = "Albert Einstein"
    assert expected_content in func(query="Albert Einstein")


def test_tool_creation_langchain_missing_imports(client: LocalClient):
    # create langchain tool
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    langchain_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Translate to memGPT Tool
    # Intentionally missing {"langchain_community.utilities": "WikipediaAPIWrapper"}
    with pytest.raises(RuntimeError):
        ToolCreate.from_langchain(langchain_tool)


def test_shared_blocks_without_send_message(client: LocalClient):
    from letta import BasicBlockMemory
    from letta.client.client import Block, create_client
    from letta.schemas.agent import AgentType
    from letta.schemas.embedding_config import EmbeddingConfig
    from letta.schemas.llm_config import LLMConfig

    client = create_client()
    shared_memory_block = Block(name="shared_memory", label="shared_memory", value="[empty]", limit=2000)
    memory = BasicBlockMemory(blocks=[shared_memory_block])

    agent_1 = client.create_agent(
        agent_type=AgentType.memgpt_agent,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        memory=memory,
    )

    agent_2 = client.create_agent(
        agent_type=AgentType.memgpt_agent,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        memory=memory,
    )

    agent_1.memory.update_block_value(label="shared_memory", value="I am no longer an [empty] memory")

    block_id = agent_1.memory.get_block("shared_memory").id
    client.update_block(block_id, text="I am no longer an [empty] memory")
    client.update_agent(agent_id=agent_1.id, memory=agent_1.memory)
    agent_1 = client.get_agent(agent_1.id)
    agent_2 = client.get_agent(agent_2.id)
    client.update_agent(agent_id=agent_2.id, memory=agent_2.memory)
    assert agent_1.memory.get_block("shared_memory").value == "I am no longer an [empty] memory"
    assert agent_2.memory.get_block("shared_memory").value == "I am no longer an [empty] memory"
