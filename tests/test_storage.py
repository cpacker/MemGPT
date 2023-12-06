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
from memgpt.connectors.chroma import ChromaStorageConnector
from memgpt.connectors.db import PostgresStorageConnector, LanceDBConnector
from memgpt.embeddings import embedding_model
from memgpt.config import MemGPTConfig, AgentConfig

import argparse


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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Missing OpenAI API key")
def test_chroma_openai():
    if not os.getenv("OPENAI_API_KEY"):
        return  # soft pass

    config = MemGPTConfig(
        archival_storage_type="chroma",
        archival_storage_path="./test_chroma",
        embedding_endpoint_type="openai",
        embedding_dim=1536,
        model="gpt4",
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
    )
    config.save()
    embed_model = embedding_model()

    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]

    db = ChromaStorageConnector(name="test-openai")

    for passage in passage:
        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))

    query = "why was she crying"
    query_vec = embed_model.get_text_embedding(query)
    res = db.query(query, query_vec, top_k=2)

    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"

    print(res[0].text)

    print("deleting")
    db.delete()


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
