from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.connectors.db import PostgresStorageConnector
from memgpt.embeddings import embedding_model
from memgpt.config import MemGPTConfig

import argparse


def test_postgres():

    config = MemGPTConfig.load()
    embed_model = embedding_model()

    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]

    db = PostgresStorageConnector(uri=config.archival_storage_uri, table_name="test2")

    for passage in passage:
        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))

    print(db.get_all())

    query = "why was she crying"
    query_vec = embed_model.get_text_embedding(query)
    res = db.query(query_vec, top_k=2)

    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"

    print("deleting")
    db.delete()


test_postgres()
