from letta.client.client import Block, create_client
from letta.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from letta.offline_memory_agent import (
    rethink_memory,
    send_message_offline_agent,
    trigger_rethink_memory,
)
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.utils import get_human_text, get_persona_text


def test_offline_memory_agent():
    client = create_client()
    assert client is not None

    trigger_rethink_memory_tool = client.create_tool(trigger_rethink_memory)
    send_message_offline_agent_tool = client.create_tool(send_message_offline_agent)

    conversation_memory = ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text(DEFAULT_PERSONA))
    new_memory = Block(name="rethink_memory_block", label="rethink_memory_block", value="", limit=2000)
    conversation_memory.link_block(new_memory)

    conversation_agent = client.create_agent(
        agent_type=AgentType.offline_memory_agent,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=[send_message_offline_agent_tool.name, trigger_rethink_memory_tool.name],
        memory=conversation_memory,
        include_base_tools=False,
    )
    assert conversation_agent is not None
    assert conversation_agent.memory.list_block_labels() == ["persona", "human", "rethink_memory_block"]

    rethink_memory_tool = client.create_tool(rethink_memory)
    offline_memory_agent = client.create_agent(
        agent_type=AgentType.offline_memory_agent,
        memory=ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text("offline_memory_persona")),
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=[rethink_memory_tool.name],
    )
    response = client.user_message(agent_id=conversation_agent.id, message="rethink: Tell me something I don't know about myself.")
    print(response)


if __name__ == "__main__":
    test_offline_memory_agent()
