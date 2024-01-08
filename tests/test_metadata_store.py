import os
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
    user_1 = User(default_llm_config=LLMConfig(model="gpt-4"))
    user_2 = User()
    agent_1 = AgentState(
        user_id=user_1.id,
        name="agent_1",
        preset=user_1.default_preset,
        persona=user_1.default_persona,
        human=user_1.default_human,
        llm_config=user_1.default_llm_config,
        embedding_config=user_1.default_embedding_config,
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
    print(agent_1.llm_config, user_1.default_llm_config)
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

    # text deletion
    ms.delete_user(user_1.id)
    ms.delete_user(user_2.id)
    ms.delete_agent(agent_1.id)
    ms.delete_source(source_1.id)
