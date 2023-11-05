import os
import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "pgvector", "psycopg", "psycopg2-binary"]
)  # , "psycopg_binary"])  # "psycopg", "libpq-dev"])
import pgvector  # Try to import again after installing

from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.connectors.db import PostgresStorageConnector
from memgpt.embeddings import embedding_model
from memgpt.config import MemGPTConfig, AgentConfig

import argparse


def test_postgres():
    config = MemGPTConfig()
    config.archival_storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")  # the URI for a postgres DB w/ the pgvector extension
    assert config.archival_storage_uri is not None
    config.archival_storage_uri = config.archival_storage_uri.replace(
        "postgres://", "postgresql://"
    )  # https://stackoverflow.com/a/64698899
    config.save()
    print(config)

    embed_model = embedding_model()

    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]

    db = PostgresStorageConnector(name="test2")

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


test_postgres()
