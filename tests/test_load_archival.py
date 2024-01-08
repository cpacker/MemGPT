import os
import uuid
import pytest
from sqlalchemy.ext.declarative import declarative_base


# import memgpt
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.cli.cli_load import load_directory, load_database, load_webpage
from memgpt.cli.cli import attach
from memgpt.config import MemGPTConfig
from memgpt.metadata import MetadataStore
from memgpt.data_types import User, AgentState, EmbeddingConfig


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
@pytest.mark.parametrize("passage_storage_connector", ["chroma", "postgres"])
def test_load_directory(metadata_storage_connector, passage_storage_connector, clear_dynamically_created_models, recreate_declarative_base):
    # setup config
    config = MemGPTConfig()
    if metadata_storage_connector == "postgres":
        if not os.getenv("PGVECTOR_TEST_DB_URL"):
            print("Skipping test, missing PG URI")
            return
        config.metadata_storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")
        config.metadata_storage_type = "postgres"
    elif metadata_storage_connector == "sqlite":
        print("testing  sqlite metadata")
        # nothing to do (should be config defaults)
    else:
        raise NotImplementedError(f"Storage type {metadata_storage_connector} not implemented")
    if passage_storage_connector == "postgres":
        if not os.getenv("PGVECTOR_TEST_DB_URL"):
            print("Skipping test, missing PG URI")
            return
        config.archival_storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")
        config.archival_storage_type = "postgres"
    elif passage_storage_connector == "chroma":
        print("testing chroma passage storage")
        # nothing to do (should be config defaults)
    else:
        raise NotImplementedError(f"Storage type {passage_storage_connector} not implemented")
    config.save()

    # create metadata store
    ms = MetadataStore(config)

    # embedding config
    if os.getenv("OPENAI_API_KEY"):
        embedding_config = EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_dim=1536,
            openai_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        embedding_config = EmbeddingConfig(embedding_endpoint_type="local", embedding_endpoint=None, embedding_dim=384)

    # create user and agent
    user = User(id=uuid.UUID(config.anon_clientid), default_embedding_config=embedding_config)
    agent = AgentState(
        user_id=user.id,
        name="test_agent",
        preset=user.default_preset,
        persona=user.default_persona,
        human=user.default_human,
        llm_config=user.default_llm_config,
        embedding_config=user.default_embedding_config,
    )
    ms.delete_user(user.id)
    ms.create_user(user)
    ms.create_agent(agent)
    user = ms.get_user(user.id)
    print("Got user:", user, user.default_embedding_config)

    # setup storage connectors
    print("Creating storage connectors...")
    user_id = user.id
    print("User ID", user_id)
    passages_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)

    # load data
    name = "test_dataset"
    cache_dir = "CONTRIBUTING.md"

    # TODO: load two different data sources

    # clear out data
    print("Resetting tables with delete_table...")
    passages_conn.delete_table()
    print("Re-creating tables...")
    passages_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
    assert passages_conn.size() == 0, f"Expected 0 records, got {passages_conn.size()}: {[vars(r) for r in passages_conn.get_all()]}"

    # test: load directory
    print("Loading directory")
    load_directory(name=name, input_dir=None, input_files=[cache_dir], recursive=False, user_id=user_id)  # cache_dir,

    # test to see if contained in storage
    print("Querying table...")
    sources = ms.list_sources(user_id=user_id)
    assert len(sources) == 1, f"Expected 1 source, but got {len(sources)}"
    assert sources[0].name == name, f"Expected name {name}, but got {sources[0].name}"
    print("Source", sources)

    # test to see if contained in storage
    assert (
        len(passages_conn.get_all()) == passages_conn.size()
    ), f"Expected {passages_conn.size()} passages, but got {len(passages_conn.get_all())}"
    passages = passages_conn.get_all({"data_source": name})
    print("Source", [p.data_source for p in passages])
    print("All sources", [p.data_source for p in passages_conn.get_all()])
    assert len(passages) > 0, f"Expected >0 passages, but got {len(passages)}"
    assert len(passages) == passages_conn.size(), f"Expected {passages_conn.size()} passages, but got {len(passages)}"
    assert [p.data_source == name for p in passages]
    print("Passages", passages)

    # test: listing sources
    print("Querying all...")
    sources = ms.list_sources(user_id=user_id)
    print("All sources", [s.name for s in sources])

    # test loading into an agent
    # create agent
    agent_id = agent.id
    # create storage connector
    print("Creating agent archival storage connector...")
    conn = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config=config, user_id=user_id, agent_id=agent_id)
    print("Deleting agent archival table...")
    conn.delete_table()
    conn = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config=config, user_id=user_id, agent_id=agent_id)
    assert conn.size() == 0, f"Expected 0 records, got {conn.size()}: {[vars(r) for r in conn.get_all()]}"

    # attach data
    print("Attaching data...")
    attach(agent=agent.name, data_source=name, user_id=user_id)

    # test to see if contained in storage
    assert len(passages) == conn.size()
    assert len(passages) == len(conn.get_all({"data_source": name}))

    # test: delete source
    passages_conn.delete({"data_source": name})
    assert len(passages_conn.get_all({"data_source": name})) == 0

    # cleanup
    ms.delete_user(user.id)
    ms.delete_agent(agent.id)
    ms.delete_source(sources[0].id)
