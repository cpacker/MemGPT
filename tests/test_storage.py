import os
import uuid
import subprocess
import sys
import pytest

# subprocess.check_call(
#    [sys.executable, "-m", "pip", "install", "pgvector", "psycopg", "psycopg2-binary"]
# )  # , "psycopg_binary"])  # "psycopg", "libpq-dev"])
#
# subprocess.check_call([sys.executable, "-m", "pip", "install", "lancedb"])
from memgpt.connectors.storage import StorageConnector, TableType
from memgpt.embeddings import embedding_model
from memgpt.data_types import Message, Passage
from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.utils import get_local_time
from memgpt.connectors.storage import StorageConnector, TableType
from memgpt.constants import DEFAULT_MEMGPT_MODEL, DEFAULT_PERSONA, DEFAULT_HUMAN

import argparse
from datetime import datetime, timedelta

# Note: the database will filter out rows that do not correspond to agent1 and test_user by default.
texts = ["This is a test passage", "This is another test passage", "Cinderella wept"]
start_date = datetime(2009, 10, 5, 18, 00)
dates = [start_date, start_date - timedelta(weeks=1), start_date + timedelta(weeks=1)]
roles = ["user", "agent", "agent"]
agent_ids = ["agent1", "agent2", "agent1"]
ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
user_id = "test_user"


# Data generation functions: Passages
def generate_passages(embed_model):
    """Generate list of 3 Passage objects"""
    # embeddings: use openai if env is set, otherwise local
    passages = []
    for text, _, _, agent_id, id in zip(texts, dates, roles, agent_ids, ids):
        embedding = None
        if embed_model:
            embedding = embed_model.get_text_embedding(text)
        passages.append(Passage(user_id=user_id, text=text, agent_id=agent_id, embedding=embedding, data_source="test_source", id=id))
    return passages


# Data generation functions: Messages
def generate_messages(embed_model):
    """Generate list of 3 Message objects"""
    messages = []
    for text, date, role, agent_id, id in zip(texts, dates, roles, agent_ids, ids):
        embedding = None
        if embed_model:
            embedding = embed_model.get_text_embedding(text)
        messages.append(
            Message(user_id=user_id, text=text, agent_id=agent_id, role=role, created_at=date, id=id, model="gpt4", embedding=embedding)
        )
        print(messages[-1].text)
    return messages


@pytest.mark.parametrize("storage_connector", ["postgres", "chroma", "sqlite"])
@pytest.mark.parametrize("table_type", [TableType.RECALL_MEMORY, TableType.ARCHIVAL_MEMORY])
def test_storage(storage_connector, table_type):
    # setup memgpt config
    # TODO: set env for different config path
    config = MemGPTConfig()
    if storage_connector == "postgres":
        if not os.getenv("PGVECTOR_TEST_DB_URL"):
            print("Skipping test, missing PG URI")
            return
        config.archival_storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")
        config.recall_storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")
        config.archival_storage_type = "postgres"
        config.recall_storage_type = "postgres"
    if storage_connector == "lancedb":
        # TODO: complete lancedb implementation
        if not os.getenv("LANCEDB_TEST_URL"):
            print("Skipping test, missing LanceDB URI")
            return
        config.archival_storage_uri = os.getenv("LANCEDB_TEST_URL")
        config.recall_storage_uri = os.getenv("LANCEDB_TEST_URL")
        config.archival_storage_type = "lancedb"
        config.recall_storage_type = "lancedb"
    if storage_connector == "chroma":
        if table_type == TableType.RECALL_MEMORY:
            print("Skipping test, chroma only supported for archival memory")
            return
        config.archival_storage_type = "chroma"
        config.archival_storage_path = "./test_chroma"
    if storage_connector == "sqlite":
        if table_type == TableType.ARCHIVAL_MEMORY:
            print("Skipping test, sqlite only supported for recall memory")
            return
        config.recall_storage_type = "local"

    # get embedding model
    embed_model = None
    if os.getenv("OPENAI_API_KEY"):
        config.embedding_endpoint_type = "openai"
        config.embedding_endpoint = "https://api.openai.com/v1"
        config.embedding_dim = 1536
        config.openai_key = os.getenv("OPENAI_API_KEY")
    else:
        config.embedding_endpoint_type = "local"
        config.embedding_endpoint = None
        config.embedding_dim = 384
    config.save()
    embed_model = embedding_model()

    # create agent
    agent_config = AgentConfig(
        name="agent1",
        persona=DEFAULT_PERSONA,
        human=DEFAULT_HUMAN,
        model=DEFAULT_MEMGPT_MODEL,
    )

    # create storage connector
    conn = StorageConnector.get_storage_connector(storage_type=storage_connector, table_type=table_type, agent_config=agent_config)
    # conn.client.delete_collection(conn.collection.name)  # clear out data
    conn.delete_table()
    conn = StorageConnector.get_storage_connector(storage_type=storage_connector, table_type=table_type, agent_config=agent_config)

    # override filters
    conn.user_id = user_id
    conn.filters = {"user_id": user_id, "agent_id": "agent1"}

    # generate data
    if table_type == TableType.ARCHIVAL_MEMORY:
        records = generate_passages(embed_model)
    elif table_type == TableType.RECALL_MEMORY:
        records = generate_messages(embed_model)
    else:
        raise NotImplementedError(f"Table type {table_type} not implemented")

    # test: insert
    conn.insert(records[0])
    assert conn.size() == 1, f"Expected 1 record, got {conn.size()}: {conn.get_all()}"

    # test: insert_many
    conn.insert_many(records[1:])
    assert (
        conn.size() == 2
    ), f"Expected 1 record, got {conn.size()}: {conn.get_all()}"  # expect 2, since storage connector filters for agent1

    # test: list_loaded_data
    # TODO: add back
    # if table_type == TableType.ARCHIVAL_MEMORY:
    #    sources = StorageConnector.list_loaded_data(storage_type=storage_connector)
    #    assert len(sources) == 1, f"Expected 1 source, got {len(sources)}"
    #    assert sources[0] == "test_source", f"Expected 'test_source', got {sources[0]}"

    # test: get_all_paginated
    paginated_total = 0
    for page in conn.get_all_paginated(page_size=1):
        paginated_total += len(page)
    assert paginated_total == 2, f"Expected 2 records, got {paginated_total}"

    # test: get_all
    all_records = conn.get_all()
    assert len(all_records) == 2, f"Expected 2 records, got {len(all_records)}"
    all_records = conn.get_all(limit=1)
    assert len(all_records) == 1, f"Expected 1 records, got {len(all_records)}"

    # test: get
    print("GET ID", ids[0], records)
    res = conn.get(id=ids[0])
    assert res.text == texts[0], f"Expected {texts[0]}, got {res.text}"

    # test: size
    assert conn.size() == 2, f"Expected 2 records, got {conn.size()}"
    assert conn.size(filters={"agent_id": "agent1"}) == 2, f"Expected 2 records, got {conn.size(filters={'agent_id', 'agent1'})}"
    if table_type == TableType.RECALL_MEMORY:
        assert conn.size(filters={"role": "user"}) == 1, f"Expected 1 record, got {conn.size(filters={'role': 'user'})}"

    # test: query (vector)
    if table_type == TableType.ARCHIVAL_MEMORY:
        query = "why was she crying"
        query_vec = embed_model.get_text_embedding(query)
        res = conn.query(None, query_vec, top_k=2)
        assert len(res) == 2, f"Expected 2 results, got {len(res)}"
        print("Archival memory results", res)
        assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"

    # test optional query functions for recall memory
    if table_type == TableType.RECALL_MEMORY:
        # test: query_text
        query = "CindereLLa"
        res = conn.query_text(query)
        assert len(res) == 1, f"Expected 1 result, got {len(res)}"
        assert "Cinderella" in res[0].text, f"Expected 'Cinderella' in results, but got {res[0].text}"

        # test: query_date (recall memory only)
        print("Testing recall memory date search")
        start_date = datetime(2009, 10, 5, 18, 00)
        start_date = start_date - timedelta(days=1)
        end_date = start_date + timedelta(days=1)
        res = conn.query_date(start_date=start_date, end_date=end_date)
        print("DATE", res)
        assert len(res) == 1, f"Expected 1 result, got {len(res)}: {res}"

    # test: delete
    conn.delete({"id": ids[0]})
    assert conn.size() == 1, f"Expected 2 records, got {conn.size()}"
