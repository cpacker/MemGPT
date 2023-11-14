import os
import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "pgvector", "psycopg", "psycopg2-binary"]
)  # , "psycopg_binary"])  # "psycopg", "libpq-dev"])
import pgvector  # Try to import again after installing

from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.connectors.db import PostgresStorageConnector, LanceDBConnector
from memgpt.embeddings import embedding_model
from memgpt.config import MemGPTConfig, AgentConfig

import argparse


def test_postgres_openai():
    assert os.getenv("PGVECTOR_TEST_DB_URL") is not None
    if os.getenv("OPENAI_API_KEY") is None:
        return  # soft pass

    # os.environ["MEMGPT_CONFIG_PATH"] = "./config"
    config = MemGPTConfig(archival_storage_type="lancedb", archival_storage_uri=os.getenv("PGVECTOR_TEST_DB_URL"))
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


def test_postgres_local():
    assert os.getenv("PGVECTOR_TEST_DB_URL") is not None
    # os.environ["MEMGPT_CONFIG_PATH"] = "./config"

    config = MemGPTConfig(
        archival_storage_type="postgres",
        archival_storage_uri=os.getenv("PGVECTOR_TEST_DB_URL"),
        embedding_model="local",
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
