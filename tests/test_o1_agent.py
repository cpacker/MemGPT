from letta.client.client import create_client
from letta.constants import DEFAULT_HUMAN
from letta.o1_agent import send_final_message, send_thinking_message
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.utils import get_human_text, get_persona_text


def test_o1_agent():
    client = create_client()
    assert client is not None

    thinking_tool = client.create_tool(send_thinking_message)
    final_tool = client.create_tool(send_final_message)

    agent_state = client.create_agent(
        agent_type=AgentType.o1_agent,
        tools=[thinking_tool.name, final_tool.name],
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        memory=ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text("o1_persona")),
    )
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    response = client.user_message(agent_id=agent_state.id, message="9.9 or 9.11, which is a larger number?")
    assert response is not None
    assert len(response.messages) > 3


if __name__ == "__main__":
    test_o1_agent()
