import os
import subprocess
import sys
import pytest

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "pgvector", "psycopg", "psycopg2-binary"]
)  # , "psycopg_binary"])  # "psycopg", "libpq-dev"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "lancedb"])
import pgvector  # Try to import again after installing

from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.connectors.db import PostgresStorageConnector, LanceDBConnector
from memgpt.embeddings import embedding_model
from memgpt.data_types import Message, Passage
from memgpt.config import MemGPTConfig, AgentConfig

import argparse


def test_recall_db() -> None:
    # os.environ["MEMGPT_CONFIG_PATH"] = "./config"

    storage_type = "postgres"
    storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")
    config = MemGPTConfig(recall_storage_type=storage_type, recall_storage_uri=storage_uri)
    print(config.config_path)
    assert config.recall_storage_uri is not None
    config.save()
    print(config)

    conn = StorageConnector.get_recall_storage_connector()

    # construct recall memory messages
    message1 = Message(
        agent_id="test_agent1",
        role="agent",
        content="This is a test message",
        id="test_id1",
    )
    message2 = Message(
        agent_id="test_agent2",
        role="user",
        content="This is a test message",
        id="test_id2",
    )

    # test insert
    conn.insert(message1)
    conn.insert_many([message2])

    # test size
    assert conn.size() == 2, f"Expected 2 messages, got {conn.size()}"
    assert conn.size(filters={"agent_id": "test_agent2"}) == 1, f"Expected 2 messages, got {conn.size()}"

    # test get
    assert conn.get("test_id1") == message1, f"Expected {message1}, got {conn.get('test_id1')}"
    assert (
        len(conn.get_all(limit=10, filters={"agent_id": "test_agent2"})) == 1
    ), f"Expected 1 message, got {len(conn.get_all(limit=10, filters={'agent_id': 'test_agent2'}))}"


@pytest.mark.skipif(not os.getenv("PGVECTOR_TEST_DB_URL") or not os.getenv("OPENAI_API_KEY"), reason="Missing PG URI and/or OpenAI API key")
def test_postgres_openai():
    if not os.getenv("PGVECTOR_TEST_DB_URL"):
        return  # soft pass
    if not os.getenv("OPENAI_API_KEY"):
        return  # soft pass

    # os.environ["MEMGPT_CONFIG_PATH"] = "./config"
    config = MemGPTConfig(archival_storage_type="postgres", archival_storage_uri=os.getenv("PGVECTOR_TEST_DB_URL"))
    print(config.config_path)
    assert config.archival_storage_uri is not None
    config.archival_storage_uri = config.archival_storage_uri.replace(
        "postgres://", "postgresql://"
    )  # https://stackoverflow.com/a/64698899
    config.save()
    print(config)

    embed_model = embedding_model()

    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]

    db = PostgresStorageConnector(name="test-openai")

    for passage in passage:
        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))

    print(db.get_all())

    query = "why was she crying"
    query_vec = embed_model.get_text_embedding(query)
    res = db.query(None, query_vec, top_k=2)

    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"

    # TODO fix (causes a hang for some reason)
    # print("deleting...")
    # db.delete()
    # print("...finished")


@pytest.mark.skipif(
    not os.getenv("LANCEDB_TEST_URL") or not os.getenv("OPENAI_API_KEY"), reason="Missing LANCEDB URI and/or OpenAI API key"
)
def test_lancedb_openai():
    assert os.getenv("LANCEDB_TEST_URL") is not None
    if os.getenv("OPENAI_API_KEY") is None:
        return  # soft pass

    config = MemGPTConfig(archival_storage_type="lancedb", archival_storage_uri=os.getenv("LANCEDB_TEST_URL"))
    print(config.config_path)
    assert config.archival_storage_uri is not None
    print(config)

    embed_model = embedding_model()

    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]

    db = LanceDBConnector(name="test-openai")

    for passage in passage:
        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))

    print(db.get_all())

    query = "why was she crying"
    query_vec = embed_model.get_text_embedding(query)
    res = db.query(None, query_vec, top_k=2)

    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"


@pytest.mark.skipif(not os.getenv("PGVECTOR_TEST_DB_URL"), reason="Missing PG URI")
def test_postgres_local():
    if not os.getenv("PGVECTOR_TEST_DB_URL"):
        return
    # os.environ["MEMGPT_CONFIG_PATH"] = "./config"

    config = MemGPTConfig(
        archival_storage_type="postgres",
        archival_storage_uri=os.getenv("PGVECTOR_TEST_DB_URL"),
        embedding_endpoint_type="local",
        embedding_dim=384,  # use HF model
    )
    print(config.config_path)
    assert config.archival_storage_uri is not None
    config.archival_storage_uri = config.archival_storage_uri.replace(
        "postgres://", "postgresql://"
    )  # https://stackoverflow.com/a/64698899
    config.save()
    print(config)

    embed_model = embedding_model()

    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]

    db = PostgresStorageConnector(name="test-local")

    for passage in passage:
        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))

    print(db.get_all())

    query = "why was she crying"
    query_vec = embed_model.get_text_embedding(query)
    res = db.query(None, query_vec, top_k=2)

    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"

    # TODO fix (causes a hang for some reason)
    # print("deleting...")
    # db.delete()
    # print("...finished")


@pytest.mark.skipif(not os.getenv("LANCEDB_TEST_URL"), reason="Missing LanceDB URI")
def test_lancedb_local():
    assert os.getenv("LANCEDB_TEST_URL") is not None

    config = MemGPTConfig(
        archival_storage_type="lancedb",
        archival_storage_uri=os.getenv("LANCEDB_TEST_URL"),
        embedding_model="local",
        embedding_dim=384,  # use HF model
    )
    print(config.config_path)
    assert config.archival_storage_uri is not None

    embed_model = embedding_model()

    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]

    db = LanceDBConnector(name="test-local")

    for passage in passage:
        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))

    print(db.get_all())

    query = "why was she crying"
    query_vec = embed_model.get_text_embedding(query)
    res = db.query(None, query_vec, top_k=2)

    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"
