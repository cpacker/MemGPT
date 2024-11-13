from letta.client.client import create_client
from letta.constants import DEFAULT_HUMAN
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.utils import get_human_text, get_persona_text


def test_offline_memory_agent():
    client = create_client()
    assert client is not None

    agent_state = client.create_agent(
        agent_type=AgentType.offline_memory_agent,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        memory=ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text("offline_memory_persona")),
    )
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    # create a interaction here

    response = client.user_message(agent_id=agent_state.id, message="")


if __name__ == "__main__":
    test_offline_memory_agent()
