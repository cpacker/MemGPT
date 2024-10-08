import json
import pytest

from letta.schemas.agent import AgentType
from letta.client.client import create_client
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig


def test_agent_creation(agent_type):
    client = create_client()
    assert client is not None

    if agent_type is not None:
        agent_state = client.create_agent(
            agent_type=agent_type,
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        )
    else:
        agent_state = client.create_agent(
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        )

    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    response = client.user_message(agent_id=agent_state.id, message="My name is Vivek.")
    assert response is not None

    print(f"Successfully created a agent of type {agent_type}!")


if __name__ == "__main__":
    # Test normal agent creation
    test_agent_creation(None)

    # Test agent creation with agent type
    test_agent_creation(AgentType.base_agent)
    with pytest.raises(NotImplementedError):
        test_agent_creation(AgentType.split_thread_agent)
