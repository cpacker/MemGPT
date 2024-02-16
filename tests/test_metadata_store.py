import os
from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PERSONA, DEFAULT_PRESET
import pytest

from memgpt.metadata import MetadataStore
from memgpt.config import MemGPTConfig
from memgpt.data_types import User, AgentState, Source, LLMConfig, EmbeddingConfig


# @pytest.mark.parametrize("storage_connector", ["postgres", "sqlite"])
@pytest.mark.parametrize("storage_connector", ["sqlite"])
def test_storage(storage_connector):
    config = MemGPTConfig()
    if storage_connector == "postgres":
        if not os.getenv("PGVECTOR_TEST_DB_URL"):
            print("Skipping test, missing PG URI")
            return
        config.archival_storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")
        config.recall_storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")
        config.archival_storage_type = "postgres"
        config.recall_storage_type = "postgres"
    if storage_connector == "sqlite":
        config.recall_storage_type = "local"

    ms = MetadataStore(config)

    # generate data
    user_1 = User()
    user_2 = User()
    agent_1 = AgentState(
        user_id=user_1.id,
        name="agent_1",
        preset=DEFAULT_PRESET,
        persona=DEFAULT_PERSONA,
        human=DEFAULT_HUMAN,
        llm_config=config.default_llm_config,
        embedding_config=config.default_embedding_config,
    )
    source_1 = Source(user_id=user_1.id, name="source_1")

    # test creation
    ms.create_user(user_1)
    ms.create_user(user_2)
    ms.create_agent(agent_1)
    ms.create_source(source_1)

    # test listing
    len(ms.list_agents(user_id=user_1.id)) == 1
    len(ms.list_agents(user_id=user_2.id)) == 0
    len(ms.list_sources(user_id=user_1.id)) == 1
    len(ms.list_sources(user_id=user_2.id)) == 0

    # test: updating

    # test: update JSON-stored LLMConfig class
    print(agent_1.llm_config, config.default_llm_config)
    llm_config = ms.get_agent(agent_1.id).llm_config
    assert isinstance(llm_config, LLMConfig), f"LLMConfig is {type(llm_config)}"
    assert llm_config.model == "gpt-4", f"LLMConfig model is {llm_config.model}"
    llm_config.model = "gpt3.5-turbo"
    agent_1.llm_config = llm_config
    ms.update_agent(agent_1)
    assert ms.get_agent(agent_1.id).llm_config.model == "gpt3.5-turbo", f"Updated LLMConfig to {ms.get_agent(agent_1.id).llm_config.model}"

    # test attaching sources
    len(ms.list_attached_sources(agent_id=agent_1.id)) == 0
    ms.attach_source(user_1.id, agent_1.id, source_1.id)
    len(ms.list_attached_sources(agent_id=agent_1.id)) == 1

    # test: detaching sources
    ms.detach_source(agent_1.id, source_1.id)
    len(ms.list_attached_sources(agent_id=agent_1.id)) == 0

    # test getting
    ms.get_user(user_1.id)
    ms.get_agent(agent_1.id)
    ms.get_source(source_1.id)

    # test api keys
    api_key = ms.create_api_key(user_id=user_1.id)
    print("api_key=", api_key.token, api_key.user_id)
    api_key_result = ms.get_api_key(api_key=api_key.token)
    assert api_key.token == api_key_result.token, (api_key, api_key_result)
    user_result = ms.get_user_from_api_key(api_key=api_key.token)
    assert user_1.id == user_result.id, (user_1, user_result)
    all_keys_for_user = ms.get_all_api_keys_for_user(user_id=user_1.id)
    assert len(all_keys_for_user) > 0, all_keys_for_user
    ms.delete_api_key(api_key=api_key.token)

    # test deletion
    ms.delete_user(user_1.id)
    ms.delete_user(user_2.id)
    ms.delete_agent(agent_1.id)
    ms.delete_source(source_1.id)
