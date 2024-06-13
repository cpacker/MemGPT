import pytest

from memgpt.settings import settings
from tests.utils import wipe_memgpt_home
from memgpt.data_types import EmbeddingConfig, LLMConfig
from memgpt.credentials import MemGPTCredentials
from memgpt.server.server import SyncServer

from tests.config import TestMGPTConfig

@pytest.fixture(scope="module")
def config():
    db_url = settings.pg_db
    use_oai = settings.openai_api_key
    config_args = {}
    arg_pairs = (
        (db_url, ("archival_storage_uri", "recall_storage_uri", "metadata_storage_uri")),
        ("postgres", ("archival_storage_type", "recall_storage_type", "metadata_storage_type")),
    )
    for arg, keys in arg_pairs:
        for key in keys:
            config_args[key] = arg

    default_embedding_config=EmbeddingConfig(
        embedding_endpoint_type="openai" if use_oai else "hugging-face",
        embedding_endpoint="https://api.openai.com/v1" if use_oai else "https://embeddings.memgpt.ai",
        embedding_model="text-embedding-ada-002" if use_oai else "BAAI/bge-large-en-v1.5",
        embedding_dim=1536 if use_oai else 1024,
    )
    default_llm_config=LLMConfig(
        model_endpoint_type="openai" if use_oai else "vllm",
        model_endpoint="https://api.openai.com/v1" if use_oai else "https://api.memgpt.ai",
        model="gpt-4" if use_oai else "ehartford/dolphin-2.5-mixtral-8x7b",
    )
    return TestMGPTConfig(
        default_embedding_config=default_embedding_config,
        default_llm_config=default_llm_config,
        **config_args,)


@pytest.fixture(scope="module")
def server(tmp_path_factory, config):
    settings.config_path = tmp_path_factory.mktemp("test") / "config"
    wipe_memgpt_home()

    if key := settings.openai_api_key:
        creds_config = {"openai_key": key}
    credentials = MemGPTCredentials(**creds_config)
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
