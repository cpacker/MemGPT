import os
import subprocess
import sys
import pytest

# subprocess.check_call(
#    [sys.executable, "-m", "pip", "install", "pgvector", "psycopg", "psycopg2-binary"]
# )  # , "psycopg_binary"])  # "psycopg", "libpq-dev"])
#
# subprocess.check_call([sys.executable, "-m", "pip", "install", "lancedb"])
import pgvector  # Try to import again after installing

from memgpt.connectors.storage import StorageConnector, TableType
from memgpt.connectors.chroma import ChromaStorageConnector
from memgpt.connectors.db import PostgresStorageConnector, LanceDBConnector
from memgpt.embeddings import embedding_model
from memgpt.data_types import Message, Passage
from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.utils import get_local_time
from memgpt.connectors.storage import StorageConnector, TableType
from memgpt.constants import DEFAULT_MEMGPT_MODEL, DEFAULT_PERSONA, DEFAULT_HUMAN

import argparse
from datetime import datetime, timedelta


texts = ["This is a test passage", "This is another test passage", "Cinderella wept"]
start_date = datetime(2009, 10, 5, 18, 00)
dates = [start_date - timedelta(weeks=1), start_date, start_date + timedelta(weeks=1)]
roles = ["user", "agent", "user"]
agent_ids = ["agent1", "agent2", "agent1"]
ids = ["test1", "test2", "test3"]  # TODO: generate unique uuid


def generate_passages(embed_model):
    """Generate list of 3 Passage objects"""
    # embeddings: use openai if env is set, otherwise local
    passages = []
    for (text, _, _, agent_id, id) in zip(texts, dates, roles, agent_ids, ids):
        embedding = None
        if embed_model:
            embedding = embed_model.get_text_embedding(text)
        passages.append(
            Passage(
                user_id="test",
                text=text,
                agent_id=agent_id,
                embedding=embedding,
                data_source="test_source",
            )
        )
    return passages


def generate_messages():
    """Generate list of 3 Message objects"""
    messages = []
    for (text, date, role, agent_id, id) in zip(texts, dates, roles, agent_ids, ids):
        messages.append(Message(user_id="test", text=text, agent_id=agent_id, role=role, created_at=date, id=id, model="gpt4"))
        print(messages[-1].text)
    return messages


@pytest.mark.parametrize("storage_connector", ["postgres", "chroma", "lancedb"])
@pytest.mark.parametrize("table_type", [TableType.ARCHIVAL_MEMORY, TableType.RECALL_MEMORY])
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
        if not os.getenv("LANCEDB_TEST_URL"):
            print("Skipping test, missing LanceDB URI")
            return
        config.archival_storage_uri = os.getenv("LANCEDB_TEST_URL")
        config.recall_storage_uri = os.getenv("LANCEDB_TEST_URL")
        config.archival_storage_type = "lancedb"
        config.recall_storage_type = "lancedb"
    if storage_connector == "chroma":
        config.archival_storage_type = "chroma"
        config.recall_storage_type = "chroma"
        config.recall_storage_path = "./test_chroma"
        config.archival_storage_path = "./test_chroma"

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

    # create agent
    agent_config = AgentConfig(
        persona=DEFAULT_PERSONA,
        human=DEFAULT_HUMAN,
        model=DEFAULT_MEMGPT_MODEL,
    )

    # create storage connector
    conn = StorageConnector.get_storage_connector(storage_type=storage_connector, table_type=table_type, agent_config=agent_config)

    # generate data
    if table_type == TableType.ARCHIVAL_MEMORY:
        records = generate_passages(embed_model)
    elif table_type == TableType.RECALL_MEMORY:
        records = generate_messages()
    else:
        raise NotImplementedError(f"Table type {table_type} not implemented")

    # test: insert
    conn.insert(records[0])
    assert conn.size() == 1, f"Expected 1 record, got {conn.size()}"

    # test: insert_many
    conn.insert_many(records[1:])
    assert conn.size() == 3, f"Expected 1 record, got {conn.size()}"

    # test: list_loaded_data
    if table_type == TableType.ARCHIVAL_MEMORY:
        sources = StorageConnector.list_loaded_data(storage_type=storage_connector)
        assert len(sources) == 1, f"Expected 1 source, got {len(sources)}"
        assert sources[0] == "test_source", f"Expected 'test_source', got {sources[0]}"

    # test: get_all_paginated
    paginated_total = 0
    for page in conn.get_all_paginated(page_size=1):
        paginated_total += len(page)
    assert paginated_total == 3, f"Expected 3 records, got {paginated_total}"

    # test: get_all
    all_records = conn.get_all()
    assert len(all_records) == 3, f"Expected 3 records, got {len(all_records)}"
    all_records = conn.get_all(limit=2)
    assert len(all_records) == 2, f"Expected 2 records, got {len(all_records)}"

    # test: get
    res = conn.get(id=ids[0])
    assert res.text == texts[0], f"Expected {texts[0]}, got {res.text}"

    # test: size
    assert conn.size() == 3, f"Expected 3 records, got {conn.size()}"
    assert conn.size(filters={"agent_id", "agent1"}) == 2, f"Expected 2 records, got {conn.size(filters={'agent_id', 'agent1'})}"
    if table_type == TableType.RECALL_MEMORY:
        assert conn.size(filters={"role": "user"}) == 1, f"Expected 1 record, got {conn.size(filters={'role': 'user'})}"

    # test: query (vector)
    if embed_model:
        query = "why was she crying"
        query_vec = embed_model.get_text_embedding(query)
        res = conn.query(None, query_vec, top_k=2)
        assert len(res) == 2, f"Expected 2 results, got {len(res)}"
        assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"

    # test: query_text
    query = "CindereLLa"
    res = conn.query_text(query)
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert "Cinderella" in res[0].text, f"Expected 'Cinderella' in results, but got {res[0].text}"

    # test: query_date (recall memory only)
    if table_type == TableType.RECALL_MEMORY:
        print("Testing recall memory date search")
        start_date = start_date - timedelta(days=1)
        end_date = start_date + timedelta(days=1)
        res = conn.query_date(start_date=start_date, end_date=end_date)
        assert len(res) == 1, f"Expected 1 result, got {len(res): {res}}"

    # test: delete
    conn.delete({"id": ids[0]})
    assert conn.size() == 2, f"Expected 2 records, got {conn.size()}"
    conn.delete()
    assert conn.size() == 0, f"Expected 0 records, got {conn.size()}"


# def test_recall_db():
#    # os.environ["MEMGPT_CONFIG_PATH"] = "./config"
#
#    storage_type = "postgres"
#    storage_uri = os.getenv("PGVECTOR_TEST_DB_URL")
#    config = MemGPTConfig(
#        recall_storage_type=storage_type,
#        recall_storage_uri=storage_uri,
#        model_endpoint_type="openai",
#        model_endpoint="https://api.openai.com/v1",
#        model="gpt4",
#    )
#    print(config.config_path)
#    assert config.recall_storage_uri is not None
#    config.save()
#    print(config)
#
#    agent_config = AgentConfig(
#        persona=config.persona,
#        human=config.human,
#        model=config.model,
#    )
#
#    conn = StorageConnector.get_recall_storage_connector(agent_config)
#
#    # construct recall memory messages
#    message1 = Message(
#        agent_id=agent_config.name,
#        role="agent",
#        text="This is a test message",
#        user_id=config.anon_clientid,
#        model=agent_config.model,
#        created_at=datetime.now(),
#    )
#    message2 = Message(
#        agent_id=agent_config.name,
#        role="user",
#        text="This is a test message",
#        user_id=config.anon_clientid,
#        model=agent_config.model,
#        created_at=datetime.now(),
#    )
#    print(vars(message1))
#
#    # test insert
#    conn.insert(message1)
#    conn.insert_many([message2])
#
#    # test size
#    assert conn.size() >= 2, f"Expected 2 messages, got {conn.size()}"
#    assert conn.size(filters={"role": "user"}) >= 1, f'Expected 2 messages, got {conn.size(filters={"role": "user"})}'
#
#    # test text query
#    res = conn.query_text("test")
#    print(res)
#    assert len(res) >= 2, f"Expected 2 messages, got {len(res)}"
#
#    # test date query
#    current_time = datetime.now()
#    ten_weeks_ago = current_time - timedelta(weeks=1)
#    res = conn.query_date(start_date=ten_weeks_ago, end_date=current_time)
#    print(res)
#    assert len(res) >= 2, f"Expected 2 messages, got {len(res)}"
#
#    print(conn.get_all())
#
#
# @pytest.mark.skipif(not os.getenv("PGVECTOR_TEST_DB_URL") or not os.getenv("OPENAI_API_KEY"), reason="Missing PG URI and/or OpenAI API key")
# def test_postgres_openai():
#    if not os.getenv("PGVECTOR_TEST_DB_URL"):
#        return  # soft pass
#    if not os.getenv("OPENAI_API_KEY"):
#        return  # soft pass
#
#    # os.environ["MEMGPT_CONFIG_PATH"] = "./config"
#    config = MemGPTConfig(archival_storage_type="postgres", archival_storage_uri=os.getenv("PGVECTOR_TEST_DB_URL"))
#    print(config.config_path)
#    assert config.archival_storage_uri is not None
#    config.archival_storage_uri = config.archival_storage_uri.replace(
#        "postgres://", "postgresql://"
#    )  # https://stackoverflow.com/a/64698899
#    config.save()
#    print(config)
#
#    embed_model = embedding_model()
#
#    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]
#
#    agent_config = AgentConfig(
#        name="test_agent",
#        persona=config.persona,
#        human=config.human,
#        model=config.model,
#    )
#
#    db = PostgresStorageConnector(agent_config=agent_config, table_type=TableType.ARCHIVAL_MEMORY)
#
#    # db.delete()
#    # return
#    for passage in passage:
#        db.insert(
#            Passage(
#                text=passage,
#                embedding=embed_model.get_text_embedding(passage),
#                user_id=config.anon_clientid,
#                agent_id="test_agent",
#                data_source="test",
#                metadata={"test_metadata_key": "test_metadata_value"},
#            )
#        )
#
#    print(db.get_all())
#
#    query = "why was she crying"
#    query_vec = embed_model.get_text_embedding(query)
#    res = db.query(None, query_vec, top_k=2)
#
#    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
#    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"
#
#    # TODO fix (causes a hang for some reason)
#    # print("deleting...")
#    # db.delete()
#    # print("...finished")
#
#
# @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Missing OpenAI API key")
# def test_chroma_openai():
#    if not os.getenv("OPENAI_API_KEY"):
#        return  # soft pass
#
#    config = MemGPTConfig(
#        archival_storage_type="chroma",
#        archival_storage_path="./test_chroma",
#        embedding_endpoint_type="openai",
#        embedding_dim=1536,
#        model="gpt4",
#        model_endpoint_type="openai",
#        model_endpoint="https://api.openai.com/v1",
#    )
#    config.save()
#    embed_model = embedding_model()
#
#    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]
#
#    db = ChromaStorageConnector(name="test-openai")
#
#    for passage in passage:
#        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))
#
#    query = "why was she crying"
#    query_vec = embed_model.get_text_embedding(query)
#    res = db.query(query, query_vec, top_k=2)
#
#    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
#    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"
#
#    print(res[0].text)
#
#    print("deleting")
#    db.delete()
#
#
# @pytest.mark.skipif(
#    not os.getenv("LANCEDB_TEST_URL") or not os.getenv("OPENAI_API_KEY"), reason="Missing LANCEDB URI and/or OpenAI API key"
# )
# def test_lancedb_openai():
#    assert os.getenv("LANCEDB_TEST_URL") is not None
#    if os.getenv("OPENAI_API_KEY") is None:
#        return  # soft pass
#
#    config = MemGPTConfig(archival_storage_type="lancedb", archival_storage_uri=os.getenv("LANCEDB_TEST_URL"))
#    print(config.config_path)
#    assert config.archival_storage_uri is not None
#    print(config)
#
#    embed_model = embedding_model()
#
#    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]
#
#    db = LanceDBConnector(name="test-openai")
#
#    for passage in passage:
#        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))
#
#    print(db.get_all())
#
#    query = "why was she crying"
#    query_vec = embed_model.get_text_embedding(query)
#    res = db.query(None, query_vec, top_k=2)
#
#    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
#    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"
#
#
# @pytest.mark.skipif(not os.getenv("PGVECTOR_TEST_DB_URL"), reason="Missing PG URI")
# def test_postgres_local():
#    if not os.getenv("PGVECTOR_TEST_DB_URL"):
#        return
#    # os.environ["MEMGPT_CONFIG_PATH"] = "./config"
#
#    config = MemGPTConfig(
#        archival_storage_type="postgres",
#        archival_storage_uri=os.getenv("PGVECTOR_TEST_DB_URL"),
#        embedding_endpoint_type="local",
#        embedding_dim=384,  # use HF model
#    )
#    print(config.config_path)
#    assert config.archival_storage_uri is not None
#    config.archival_storage_uri = config.archival_storage_uri.replace(
#        "postgres://", "postgresql://"
#    )  # https://stackoverflow.com/a/64698899
#    config.save()
#    print(config)
#
#    embed_model = embedding_model()
#
#    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]
#
#    db = PostgresStorageConnector(name="test-local")
#
#    for passage in passage:
#        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))
#
#    print(db.get_all())
#
#    query = "why was she crying"
#    query_vec = embed_model.get_text_embedding(query)
#    res = db.query(None, query_vec, top_k=2)
#
#    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
#    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"
#
#    # TODO fix (causes a hang for some reason)
#    # print("deleting...")
#    # db.delete()
#    # print("...finished")
#
#
# @pytest.mark.skipif(not os.getenv("LANCEDB_TEST_URL"), reason="Missing LanceDB URI")
# def test_lancedb_local():
#    assert os.getenv("LANCEDB_TEST_URL") is not None
#
#    config = MemGPTConfig(
#        archival_storage_type="lancedb",
#        archival_storage_uri=os.getenv("LANCEDB_TEST_URL"),
#        embedding_model="local",
#        embedding_dim=384,  # use HF model
#    )
#    print(config.config_path)
#    assert config.archival_storage_uri is not None
#
#    embed_model = embedding_model()
#
#    passage = ["This is a test passage", "This is another test passage", "Cinderella wept"]
#
#    db = LanceDBConnector(name="test-local")
#
#    for passage in passage:
#        db.insert(Passage(text=passage, embedding=embed_model.get_text_embedding(passage)))
#
#    print(db.get_all())
#
#    query = "why was she crying"
#    query_vec = embed_model.get_text_embedding(query)
#    res = db.query(None, query_vec, top_k=2)
#
#    assert len(res) == 2, f"Expected 2 results, got {len(res)}"
#    assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"
#
