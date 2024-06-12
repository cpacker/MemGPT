import pytest

from memgpt.settings import settings
from tests.utils import wipe_memgpt_home
from memgpt.data_types import EmbeddingConfig, LLMConfig
from memgpt.credentials import MemGPTCredentials
from memgpt.server.server import SyncServer

from tests.config import TestMGPTConfig

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    settings.config_path = tmp_path_factory.mktemp("test") / "config"
    wipe_memgpt_home()

    db_url = settings.pg_db # start of the conftest hook here

    if settings.openai_api_key:
        config = TestMGPTConfig(
            archival_storage_uri=db_url,
            recall_storage_uri=db_url,
            metadata_storage_uri=db_url,
            archival_storage_type="postgres",
            recall_storage_type="postgres",
            metadata_storage_type="postgres",
            # embeddings
            default_embedding_config=EmbeddingConfig(
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_model="text-embedding-ada-002",
                embedding_dim=1536,
            ),
            # llms
            default_llm_config=LLMConfig(
                model_endpoint_type="openai",
                model_endpoint="https://api.openai.com/v1",
                model="gpt-4",
            ),
        )
        credentials = MemGPTCredentials(
            openai_key=os.getenv("OPENAI_API_KEY"),
        )
    else:  # hosted
        config = TestMGPTConfig(
            archival_storage_uri=db_url,
            recall_storage_uri=db_url,
            metadata_storage_uri=db_url,
            archival_storage_type="postgres",
            recall_storage_type="postgres",
            metadata_storage_type="postgres",
            # embeddings
            default_embedding_config=EmbeddingConfig(
                embedding_endpoint_type="hugging-face",
                embedding_endpoint="https://embeddings.memgpt.ai",
                embedding_model="BAAI/bge-large-en-v1.5",
                embedding_dim=1024,
            ),
            # llms
            default_llm_config=LLMConfig(
                model_endpoint_type="vllm",
                model_endpoint="https://api.memgpt.ai",
                model="ehartford/dolphin-2.5-mixtral-8x7b",
            ),
        )
        credentials = MemGPTCredentials()

    config.save()
    credentials.save()
    server = SyncServer(config=config)
    return server


@pytest.fixture(scope="module")
def user_id(server):
    # create user
    user = server.create_user()
    print(f"Created user\n{user.id}")

    # initialize with default presets
    server.initialize_default_presets(user.id)
    yield user.id

    # cleanup
    server.delete_user(user.id)


@pytest.fixture(scope="module")
def agent_id(server, user_id):
    # create agent
    agent_state = server.create_agent(
        user_id=user_id,
        name="test_agent",
        preset="memgpt_chat",
    )
    print(f"Created agent\n{agent_state}")
    yield agent_state.id

    # cleanup
    server.delete_agent(user_id, agent_state.id)
