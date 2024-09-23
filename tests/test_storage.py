# TODO: add back post DB refactor

# import os
# import uuid
# from datetime import datetime, timedelta
#
# import pytest
# from sqlalchemy.ext.declarative import declarative_base
#
# from letta.agent_store.storage import StorageConnector, TableType
# from letta.config import LettaConfig
# from letta.constants import BASE_TOOLS, MAX_EMBEDDING_DIM
# from letta.credentials import LettaCredentials
# from letta.embeddings import embedding_model, query_embedding
# from letta.metadata import MetadataStore
# from letta.settings import settings
# from tests import TEST_MEMGPT_CONFIG
# from tests.utils import create_config, wipe_config
#
# from .utils import with_qdrant_storage
#
# from letta.schemas.agent import AgentState
# from letta.schemas.message import Message
# from letta.schemas.passage import Passage
# from letta.schemas.user import User
#
#
## Note: the database will filter out rows that do not correspond to agent1 and test_user by default.
# texts = ["This is a test passage", "This is another test passage", "Cinderella wept"]
# start_date = datetime(2009, 10, 5, 18, 00)
# dates = [start_date, start_date - timedelta(weeks=1), start_date + timedelta(weeks=1)]
# roles = ["user", "assistant", "assistant"]
# agent_1_id = uuid.uuid4()
# agent_2_id = uuid.uuid4()
# agent_ids = [agent_1_id, agent_2_id, agent_1_id]
# ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
# user_id = uuid.uuid4()
#
#
## Data generation functions: Passages
# def generate_passages(embed_model):
#    """Generate list of 3 Passage objects"""
#    # embeddings: use openai if env is set, otherwise local
#    passages = []
#    for text, _, _, agent_id, id in zip(texts, dates, roles, agent_ids, ids):
#        embedding, embedding_model, embedding_dim = None, None, None
#        if embed_model:
#            embedding = embed_model.get_text_embedding(text)
#            embedding_model = "gpt-4"
#            embedding_dim = len(embedding)
#        passages.append(
#            Passage(
#                user_id=user_id,
#                text=text,
#                agent_id=agent_id,
#                embedding=embedding,
#                data_source="test_source",
#                id=id,
#                embedding_dim=embedding_dim,
#                embedding_model=embedding_model,
#            )
#        )
#    return passages
#
#
## Data generation functions: Messages
# def generate_messages(embed_model):
#    """Generate list of 3 Message objects"""
#    messages = []
#    for text, date, role, agent_id, id in zip(texts, dates, roles, agent_ids, ids):
#        embedding, embedding_model, embedding_dim = None, None, None
#        if embed_model:
#            embedding = embed_model.get_text_embedding(text)
#            embedding_model = "gpt-4"
#            embedding_dim = len(embedding)
#        messages.append(
#            Message(
#                user_id=user_id,
#                text=text,
#                agent_id=agent_id,
#                role=role,
#                created_at=date,
#                id=id,
#                model="gpt-4",
#                embedding=embedding,
#                embedding_model=embedding_model,
#                embedding_dim=embedding_dim,
#            )
#        )
#        print(messages[-1].text)
#    return messages
#
#
# @pytest.fixture(autouse=True)
# def clear_dynamically_created_models():
#    """Wipe globals for SQLAlchemy"""
#    yield
#    for key in list(globals().keys()):
#        if key.endswith("Model"):
#            del globals()[key]
#
#
# @pytest.fixture(autouse=True)
# def recreate_declarative_base():
#    """Recreate the declarative base before each test"""
#    global Base
#    Base = declarative_base()
#    yield
#    Base.metadata.clear()
#
#
# @pytest.mark.parametrize("storage_connector", with_qdrant_storage(["postgres", "chroma", "sqlite", "milvus"]))
## @pytest.mark.parametrize("storage_connector", ["sqlite", "chroma"])
## @pytest.mark.parametrize("storage_connector", ["postgres"])
# @pytest.mark.parametrize("table_type", [TableType.RECALL_MEMORY, TableType.ARCHIVAL_MEMORY])
# def test_storage(
#    storage_connector,
#    table_type,
#    clear_dynamically_created_models,
#    recreate_declarative_base,
# ):
#    # setup letta config
#    # TODO: set env for different config path
#
#    # hacky way to cleanup globals that scruw up tests
#    # for table_name in ['Message']:
#    #    if 'Message' in globals():
#    #        print("Removing messages", globals()['Message'])
#    #        del globals()['Message']
#
#    wipe_config()
#    if os.getenv("OPENAI_API_KEY"):
#        create_config("openai")
#        credentials = LettaCredentials(
#            openai_key=os.getenv("OPENAI_API_KEY"),
#        )
#    else:  # hosted
#        create_config("letta_hosted")
#        LettaCredentials()
#
#    config = LettaConfig.load()
#    TEST_MEMGPT_CONFIG.default_embedding_config = config.default_embedding_config
#    TEST_MEMGPT_CONFIG.default_llm_config = config.default_llm_config
#
#    if storage_connector == "postgres":
#        TEST_MEMGPT_CONFIG.archival_storage_uri = settings.letta_pg_uri
#        TEST_MEMGPT_CONFIG.recall_storage_uri = settings.letta_pg_uri
#        TEST_MEMGPT_CONFIG.archival_storage_type = "postgres"
#        TEST_MEMGPT_CONFIG.recall_storage_type = "postgres"
#    if storage_connector == "lancedb":
#        # TODO: complete lancedb implementation
#        if not os.getenv("LANCEDB_TEST_URL"):
#            print("Skipping test, missing LanceDB URI")
#            return
#        TEST_MEMGPT_CONFIG.archival_storage_uri = os.environ["LANCEDB_TEST_URL"]
#        TEST_MEMGPT_CONFIG.recall_storage_uri = os.environ["LANCEDB_TEST_URL"]
#        TEST_MEMGPT_CONFIG.archival_storage_type = "lancedb"
#        TEST_MEMGPT_CONFIG.recall_storage_type = "lancedb"
#    if storage_connector == "chroma":
#        if table_type == TableType.RECALL_MEMORY:
#            print("Skipping test, chroma only supported for archival memory")
#            return
#        TEST_MEMGPT_CONFIG.archival_storage_type = "chroma"
#        TEST_MEMGPT_CONFIG.archival_storage_path = "./test_chroma"
#    if storage_connector == "sqlite":
#        if table_type == TableType.ARCHIVAL_MEMORY:
#            print("Skipping test, sqlite only supported for recall memory")
#            return
#        TEST_MEMGPT_CONFIG.recall_storage_type = "sqlite"
#    if storage_connector == "qdrant":
#        if table_type == TableType.RECALL_MEMORY:
#            print("Skipping test, Qdrant only supports archival memory")
#            return
#        TEST_MEMGPT_CONFIG.archival_storage_type = "qdrant"
#        TEST_MEMGPT_CONFIG.archival_storage_uri = "localhost:6333"
#    if storage_connector == "milvus":
#        if table_type == TableType.RECALL_MEMORY:
#            print("Skipping test, Milvus only supports archival memory")
#            return
#        TEST_MEMGPT_CONFIG.archival_storage_type = "milvus"
#        TEST_MEMGPT_CONFIG.archival_storage_uri = "./milvus.db"
#    # get embedding model
#    embedding_config = TEST_MEMGPT_CONFIG.default_embedding_config
#    embed_model = embedding_model(TEST_MEMGPT_CONFIG.default_embedding_config)
#
#    # create user
#    ms = MetadataStore(TEST_MEMGPT_CONFIG)
#    ms.delete_user(user_id)
#    user = User(id=user_id)
#    agent = AgentState(
#        user_id=user_id,
#        name="agent_1",
#        id=agent_1_id,
#        llm_config=TEST_MEMGPT_CONFIG.default_llm_config,
#        embedding_config=TEST_MEMGPT_CONFIG.default_embedding_config,
#        system="",
#        tools=BASE_TOOLS,
#        state={
#            "persona": "",
#            "human": "",
#            "messages": None,
#        },
#    )
#    ms.create_user(user)
#    ms.create_agent(agent)
#
#    # create storage connector
#    conn = StorageConnector.get_storage_connector(table_type, config=TEST_MEMGPT_CONFIG, user_id=user_id, agent_id=agent.id)
#    # conn.client.delete_collection(conn.collection.name)  # clear out data
#    conn.delete_table()
#    conn = StorageConnector.get_storage_connector(table_type, config=TEST_MEMGPT_CONFIG, user_id=user_id, agent_id=agent.id)
#
#    # generate data
#    if table_type == TableType.ARCHIVAL_MEMORY:
#        records = generate_passages(embed_model)
#    elif table_type == TableType.RECALL_MEMORY:
#        records = generate_messages(embed_model)
#    else:
#        raise NotImplementedError(f"Table type {table_type} not implemented")
#
#    # check record dimentions
#    print("TABLE TYPE", table_type, type(records[0]), len(records[0].embedding))
#    if embed_model:
#        assert len(records[0].embedding) == MAX_EMBEDDING_DIM, f"Expected {MAX_EMBEDDING_DIM}, got {len(records[0].embedding)}"
#        assert (
#            records[0].embedding_dim == embedding_config.embedding_dim
#        ), f"Expected {embedding_config.embedding_dim}, got {records[0].embedding_dim}"
#
#    # test: insert
#    conn.insert(records[0])
#    assert conn.size() == 1, f"Expected 1 record, got {conn.size()}: {conn.get_all()}"
#
#    # test: insert_many
#    conn.insert_many(records[1:])
#    assert (
#        conn.size() == 2
#    ), f"Expected 2 records, got {conn.size()}: {conn.get_all()}"  # expect 2, since storage connector filters for agent1
#
#    # test: update
#    # NOTE: only testing with messages
#    if table_type == TableType.RECALL_MEMORY:
#        TEST_STRING = "hello world"
#
#        updated_record = records[1]
#        updated_record.text = TEST_STRING
#
#        current_record = conn.get(id=updated_record.id)
#        assert current_record is not None, f"Couldn't find {updated_record.id}"
#        assert current_record.text != TEST_STRING, (current_record.text, TEST_STRING)
#
#        conn.update(updated_record)
#        new_record = conn.get(id=updated_record.id)
#        assert new_record is not None, f"Couldn't find {updated_record.id}"
#        assert new_record.text == TEST_STRING, (new_record.text, TEST_STRING)
#
#    # test: list_loaded_data
#    # TODO: add back
#    # if table_type == TableType.ARCHIVAL_MEMORY:
#    #    sources = StorageConnector.list_loaded_data(storage_type=storage_connector)
#    #    assert len(sources) == 1, f"Expected 1 source, got {len(sources)}"
#    #    assert sources[0] == "test_source", f"Expected 'test_source', got {sources[0]}"
#
#    # test: get_all_paginated
#    paginated_total = 0
#    for page in conn.get_all_paginated(page_size=1):
#        paginated_total += len(page)
#    assert paginated_total == 2, f"Expected 2 records, got {paginated_total}"
#
#    # test: get_all
#    all_records = conn.get_all()
#    assert len(all_records) == 2, f"Expected 2 records, got {len(all_records)}"
#    all_records = conn.get_all(limit=1)
#    assert len(all_records) == 1, f"Expected 1 records, got {len(all_records)}"
#
#    # test: get
#    print("GET ID", ids[0], records)
#    res = conn.get(id=ids[0])
#    assert res.text == texts[0], f"Expected {texts[0]}, got {res.text}"
#
#    # test: size
#    assert conn.size() == 2, f"Expected 2 records, got {conn.size()}"
#    assert conn.size(filters={"agent_id": agent.id}) == 2, f"Expected 2 records, got {conn.size(filters={'agent_id', agent.id})}"
#    if table_type == TableType.RECALL_MEMORY:
#        assert conn.size(filters={"role": "user"}) == 1, f"Expected 1 record, got {conn.size(filters={'role': 'user'})}"
#
#    # test: query (vector)
#    if table_type == TableType.ARCHIVAL_MEMORY:
#        query = "why was she crying"
#        query_vec = query_embedding(embed_model, query)
#        res = conn.query(None, query_vec, top_k=2)
#        assert len(res) == 2, f"Expected 2 results, got {len(res)}"
#        print("Archival memory results", res)
#        assert "wept" in res[0].text, f"Expected 'wept' in results, but got {res[0].text}"
#
#    # test optional query functions for recall memory
#    if table_type == TableType.RECALL_MEMORY:
#        # test: query_text
#        query = "CindereLLa"
#        res = conn.query_text(query)
#        assert len(res) == 1, f"Expected 1 result, got {len(res)}"
#        assert "Cinderella" in res[0].text, f"Expected 'Cinderella' in results, but got {res[0].text}"
#
#        # test: query_date (recall memory only)
#        print("Testing recall memory date search")
#        start_date = datetime(2009, 10, 5, 18, 00)
#        start_date = start_date - timedelta(days=1)
#        end_date = start_date + timedelta(days=1)
#        res = conn.query_date(start_date=start_date, end_date=end_date)
#        print("DATE", res)
#        assert len(res) == 1, f"Expected 1 result, got {len(res)}: {res}"
#
#    # test: delete
#    conn.delete({"id": ids[0]})
#    assert conn.size() == 1, f"Expected 2 records, got {conn.size()}"
#
#    # cleanup
#    ms.delete_user(user_id)
#
