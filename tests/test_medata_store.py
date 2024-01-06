from memgpt.metadata import MetadataStore
from memgpt.config import MemGPTConfig
from memgpt.data_types import User, AgentState, Source, LLMConfig, EmbeddingConfig


@pytest.mark.parametrize("storage_connector", ["postgres", "sqlite"])
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

    ms = MetadataStore()

    # generate data
    user_1 = User()
    user_2 = User()
    agent_1 = AgentState(user_id=user_1.id, name="agent_1")
    source_1 = Source(user_id=user_1.id, name="source_1")

    # test creation
    ms.create_user(user_1)
    ms.create_user(user_2)
    ms.create_agent(agent_1)
    ms.create_source(source_1)

    # test listing
    len(ms.list_users()) == 2

    # test getting
    ms.get_user(user_1.id)
    ms.get_agent(agent_1.id)
    ms.get_source(source_1.id)

    # text deletion
    ms.delete_user(user_1.id)
    ms.delete_user(user_2.id)
    ms.delete_agent(agent_1.id)
    ms.delete_source(source_1.id)
