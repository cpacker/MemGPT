from letta import BasicBlockMemory
from letta.client.client import Block, create_client
from letta.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from letta.offline_memory_agent import (
    finish_rethinking_memory,
    rethink_memory,
    send_message_offline_agent,
    trigger_rethink_memory,
)
from letta.prompts import gpt_system
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.tool_rule import TerminalToolRule
from letta.utils import get_human_text, get_persona_text


def test_offline_memory_agent():
    client = create_client()
    assert client is not None

    trigger_rethink_memory_tool = client.create_tool(trigger_rethink_memory)
    send_message_offline_agent_tool = client.create_tool(send_message_offline_agent)

    conversation_human_block = Block(name="human", label="human", value=get_human_text(DEFAULT_HUMAN), limit=2000)
    conversation_persona_block = Block(name="persona", label="persona", value=get_persona_text(DEFAULT_PERSONA), limit=2000)
    offline_human_block = Block(name="human", label="human", value=get_human_text(DEFAULT_HUMAN), limit=2000)
    offline_persona_block = Block(name="persona", label="persona", value=get_persona_text("offline_memory_persona"), limit=2000)

    block1 = Block(
        name="interaction_1",
        label="interaction_1",
        value="User clicked on product 2, and not product 1. User clicked on product 1 not product 3.",
        limit=2000,
    )
    block2 = Block(
        name="interaction_2",
        label="interaction_2",
        value="User clicked on product 2 and not product 3. User clicked on product 1 not product 3.",
        limit=2000,
    )

    new_memory = Block(name="rethink_memory_block", label="rethink_memory_block", value="[empty]", limit=2000)
    conversation_memory = BasicBlockMemory(blocks=[conversation_persona_block, conversation_human_block, block1, block2, new_memory])
    offline_memory = BasicBlockMemory(blocks=[offline_persona_block, offline_human_block, block1, block2, new_memory])

    conversation_agent = client.create_agent(
        agent_type=AgentType.memgpt_agent,
        system=gpt_system.get_system_text("memgpt_convo_only"),
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=[send_message_offline_agent_tool.name, trigger_rethink_memory_tool.name],
        memory=conversation_memory,
        include_base_tools=False,
    )
    assert conversation_agent is not None
    assert conversation_agent.memory.list_block_labels() == ["persona", "human", "interaction_1", "interaction_2", "rethink_memory_block"]

    rethink_memory_tool = client.create_tool(rethink_memory)
    finish_rethinking_memory_tool = client.create_tool(finish_rethinking_memory)
    offline_memory_agent = client.create_agent(
        agent_type=AgentType.offline_memory_agent,
        system=gpt_system.get_system_text("memgpt_offline_memory"),
        memory=offline_memory,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=[rethink_memory_tool.name, finish_rethinking_memory_tool.name],
        tool_rules=[TerminalToolRule(tool_name=finish_rethinking_memory_tool.name)],
        include_base_tools=False,
    )
    assert offline_memory_agent is not None
    assert offline_memory_agent.memory.list_block_labels() == ["persona", "human", "interaction_1", "interaction_2", "rethink_memory_block"]

    _ = client.user_message(agent_id=conversation_agent.id, message="trigger_rethink_memory")

    offline_memory_agent = client.get_agent(agent_id=offline_memory_agent.id)
    _ = client.user_message(agent_id=offline_memory_agent.id, message="I'm done talking now!")
    offline_memory_agent = client.get_agent(agent_id=offline_memory_agent.id)
    assert offline_memory_agent.memory.get_block("rethink_memory_block").value != "[empty]"

    _ = client.user_message(agent_id=conversation_agent.id, message="I'm done talking now!")
    conversation_agent = client.get_agent(agent_id=conversation_agent.id)
    assert conversation_agent.memory.get_block("rethink_memory_block").value != "[empty]"


if __name__ == "__main__":
    test_offline_memory_agent()
