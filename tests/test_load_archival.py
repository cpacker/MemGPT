import os
import uuid

import pytest
from sqlalchemy.ext.declarative import declarative_base

from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.cli.cli_load import load_directory
from memgpt.config import MemGPTConfig
from memgpt.credentials import MemGPTCredentials
from memgpt.data_types import EmbeddingConfig, User
from memgpt.metadata import MetadataStore

# from memgpt.data_sources.connectors import DirectoryConnector, load_data
# import memgpt
from memgpt.settings import settings
from tests import TEST_MEMGPT_CONFIG

from .utils import create_config, wipe_config, with_qdrant_storage

GET_ALL_LIMIT = 1000


@pytest.fixture(autouse=True)
def clear_dynamically_created_models():
    """Wipe globals for SQLAlchemy"""
    yield
    for key in list(globals().keys()):
        if key.endswith("Model"):
            del globals()[key]


@pytest.fixture(autouse=True)
def recreate_declarative_base():
    """Recreate the declarative base before each test"""
    global Base
    Base = declarative_base()
    yield
    Base.metadata.clear()


@pytest.mark.parametrize("metadata_storage_connector", ["sqlite", "postgres"])
@pytest.mark.parametrize("passage_storage_connector", with_qdrant_storage(["chroma", "postgres", "milvus"]))
def test_load_directory(
    metadata_storage_connector,
    passage_storage_connector,
    clear_dynamically_created_models,
    recreate_declarative_base,
):
    wipe_config()
    if os.getenv("OPENAI_API_KEY"):
        create_config("openai")
        credentials = MemGPTCredentials(
            openai_key=os.getenv("OPENAI_API_KEY"),
        )
    else:  # hosted
        create_config("memgpt_hosted")
        credentials = MemGPTCredentials()

    config = MemGPTConfig.load()
    TEST_MEMGPT_CONFIG.default_embedding_config = config.default_embedding_config
    TEST_MEMGPT_CONFIG.default_llm_config = config.default_llm_config

    # setup config
    if metadata_storage_connector == "postgres":
        TEST_MEMGPT_CONFIG.metadata_storage_uri = settings.memgpt_pg_uri
        TEST_MEMGPT_CONFIG.metadata_storage_type = "postgres"
    elif metadata_storage_connector == "sqlite":
        print("testing  sqlite metadata")
        # nothing to do (should be config defaults)
    else:
        raise NotImplementedError(f"Storage type {metadata_storage_connector} not implemented")
    if passage_storage_connector == "postgres":
        TEST_MEMGPT_CONFIG.archival_storage_uri = settings.memgpt_pg_uri
        TEST_MEMGPT_CONFIG.archival_storage_type = "postgres"
    elif passage_storage_connector == "chroma":
        print("testing chroma passage storage")
        # nothing to do (should be config defaults)
    elif passage_storage_connector == "qdrant":
        print("Testing Qdrant passage storage")
        TEST_MEMGPT_CONFIG.archival_storage_type = "qdrant"
        TEST_MEMGPT_CONFIG.archival_storage_uri = "localhost:6333"
    elif passage_storage_connector == "milvus":
        print("Testing Milvus passage storage")
        TEST_MEMGPT_CONFIG.archival_storage_type = "milvus"
        TEST_MEMGPT_CONFIG.archival_storage_uri = "./milvus.db"
    else:
        raise NotImplementedError(f"Storage type {passage_storage_connector} not implemented")
    TEST_MEMGPT_CONFIG.save()

    # create metadata store
    ms = MetadataStore(TEST_MEMGPT_CONFIG)
    user = User(id=uuid.UUID(TEST_MEMGPT_CONFIG.anon_clientid))

    # embedding config
    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI embeddings for testing")
        credentials = MemGPTCredentials(
            openai_key=os.getenv("OPENAI_API_KEY"),
        )
        credentials.save()
        embedding_config = EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_dim=1536,
            embedding_model="text-embedding-ada-002",
            # openai_key=os.getenv("OPENAI_API_KEY"),
        )

    else:
        # print("Using local embedding model for testing")
        # embedding_config = EmbeddingConfig(
        #     embedding_endpoint_type="local",
        #     embedding_endpoint=None,
        #     embedding_dim=384,
        # )

        print("Using official hosted embedding model for testing")
        embedding_config = EmbeddingConfig(
            embedding_endpoint_type="hugging-face",
            embedding_endpoint="https://embeddings.memgpt.ai",
            embedding_model="BAAI/bge-large-en-v1.5",
            embedding_dim=1024,
        )

    # write out the config so that the 'load' command will use it (CLI commands pull from config)
    TEST_MEMGPT_CONFIG.default_embedding_config = embedding_config
    TEST_MEMGPT_CONFIG.save()
    # config.default_embedding_config = embedding_config
    # config.save()

    # create user and agent
    # agent = AgentState(
    #    user_id=user.id,
    #    name="test_agent",
    #    preset=TEST_MEMGPT_CONFIG.preset,
    #    persona=get_persona_text(TEST_MEMGPT_CONFIG.persona),
    #    human=get_human_text(TEST_MEMGPT_CONFIG.human),
    #    llm_config=TEST_MEMGPT_CONFIG.default_llm_config,
    #    embedding_config=TEST_MEMGPT_CONFIG.default_embedding_config,
    #    tools=[],
    #    system="",
    # )
    ms.delete_user(user.id)
    ms.create_user(user)
    # ms.create_agent(agent)
    user = ms.get_user(user.id)
    print("Got user:", user, embedding_config)

    # setup storage connectors
    print("Creating storage connectors...")
    user_id = user.id
    print("User ID", user_id)
    passages_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, TEST_MEMGPT_CONFIG, user_id)

    # load data
    name = "test_dataset"
    cache_dir = "CONTRIBUTING.md"

    # TODO: load two different data sources

    # clear out data
    print("Resetting tables with delete_table...")
    passages_conn.delete_table()
    print("Re-creating tables...")
    passages_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, TEST_MEMGPT_CONFIG, user_id)
    assert (
        passages_conn.size() == 0
    ), f"Expected 0 records, got {passages_conn.size()}: {[vars(r) for r in passages_conn.get_all(limit=GET_ALL_LIMIT)]}"

    # test: load directory
    print("Loading directory")
    # load_directory(name=name, input_dir=None, input_files=[cache_dir], recursive=False, user_id=user_id)  # cache_dir,
    load_directory(name=name, input_files=[cache_dir], recursive=False, user_id=user_id)  # cache_dir,

    # test to see if contained in storage
    print("Querying table...")
    sources = ms.list_sources(user_id=user_id)
    assert len(sources) == 1, f"Expected 1 source, but got {len(sources)}"
    assert sources[0].name == name, f"Expected name {name}, but got {sources[0].name}"
    print("Source", sources)

    # test to see if contained in storage
    assert (
        len(passages_conn.get_all(limit=GET_ALL_LIMIT)) == passages_conn.size()
    ), f"Expected {passages_conn.size()} passages, but got {len(passages_conn.get_all(limit=GET_ALL_LIMIT))}"
    passages = passages_conn.get_all({"data_source": name}, limit=GET_ALL_LIMIT)
    print("Source", [p.data_source for p in passages])
    print(passages_conn.get_all(limit=GET_ALL_LIMIT))
    print("All sources", [p.data_source for p in passages_conn.get_all(limit=GET_ALL_LIMIT)])
    assert len(passages) > 0, f"Expected >0 passages, but got {len(passages)}"
    assert len(passages) == passages_conn.size(), f"Expected {passages_conn.size()} passages, but got {len(passages)}"
    assert [p.data_source == name for p in passages]
    print("Passages", passages)

    # test: listing sources
    print("Querying all...")
    sources = ms.list_sources(user_id=user_id)
    print("All sources", [s.name for s in sources])

    # TODO: add back once agent attachment fully supported from server
    ## test loading into an agent
    ## create agent
    # agent_id = agent.id
    ## create storage connector
    # print("Creating agent archival storage connector...")
    # conn = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config=config, user_id=user_id, agent_id=agent_id)
    # print("Deleting agent archival table...")
    # conn.delete_table()
    # conn = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config=config, user_id=user_id, agent_id=agent_id)
    # assert conn.size() == 0, f"Expected 0 records, got {conn.size()}: {[vars(r) for r in conn.get_all(limit=GET_ALL_LIMIT)]}"

    ## attach data
    # print("Attaching data...")
    # attach(agent_name=agent.name, data_source=name, user_id=user_id)

    ## test to see if contained in storage
    # assert len(passages) == conn.size()
    # assert len(passages) == len(conn.get_all({"data_source": name}, limit=GET_ALL_LIMIT))

    ## test: delete source
    # passages_conn.delete({"data_source": name})
    # assert len(passages_conn.get_all({"data_source": name}, limit=GET_ALL_LIMIT)) == 0

    # cleanup
    ms.delete_user(user.id)
    ms.delete_source(sources[0].id)

    # revert to openai config
    # client = MemGPT(quickstart="openai", user_id=user.id)
    wipe_config()
