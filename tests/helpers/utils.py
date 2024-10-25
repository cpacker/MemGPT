from typing import Optional, Union

from fastapi.testclient import TestClient

from letta import LocalClient, RESTClient
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig


def cleanup(client: Union[LocalClient, RESTClient], agent_uuid: str):
    # Clear all agents
    for agent_state in client.list_agents():
        if agent_state.name == agent_uuid:
            client.delete_agent(agent_id=agent_state.id)
            print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")

def set_default_configs(letta_client: Union[LocalClient, RESTClient], mock_llm_client: Optional[TestClient] = None):
    # Conditionally set configs based on provided llm api option
    configs_response = mock_llm_client.get("/configs") if mock_llm_client else None
    if configs_response and configs_response.status_code == 200:
        configs = configs_response.json()
        llm_config = LLMConfig.parse_obj(configs["llm"])
        embedding_config = EmbeddingConfig.parse_obj(configs["embedding"])
    else:
        llm_config = LLMConfig.default_config("gpt-4")
        embedding_config = EmbeddingConfig.default_config(provider="openai")
    
    letta_client.set_default_llm_config(llm_config)
    letta_client.set_default_embedding_config(embedding_config)
