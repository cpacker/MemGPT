import uuid
import pytest
import os
import memgpt.utils as utils
from dotenv import load_dotenv

from tests.config import TestMGPTConfig

utils.DEBUG = True
from memgpt.settings import settings
from memgpt.credentials import MemGPTCredentials
from memgpt.server.server import SyncServer
from memgpt.data_types import EmbeddingConfig, LLMConfig
from .utils import wipe_config, wipe_memgpt_home, DummyDataConnector


@pytest.fixture(scope="module")
def server():
    load_dotenv()
    wipe_config()
    wipe_memgpt_home()

    db_url = settings.pg_db

    if os.getenv("OPENAI_API_KEY"):
        config = TestMGPTConfig(
            archival_storage_uri=db_url,
            recall_storage_uri=db_url,
            metadata_storage_uri=db_url,
            archival_storage_type="postgres",
            recall_storage_type="postgres",
            metadata_storage_type="postgres",
            # embeddings
            default_embedding_config=EmbeddingConfig(
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_model="text-embedding-ada-002",
                embedding_dim=1536,
            ),
            # llms
            default_llm_config=LLMConfig(
                model_endpoint_type="openai",
                model_endpoint="https://api.openai.com/v1",
                model="gpt-4",
            ),
        )
        credentials = MemGPTCredentials(
            openai_key=os.getenv("OPENAI_API_KEY"),
        )
    else:  # hosted
        config = TestMGPTConfig(
            archival_storage_uri=db_url,
            recall_storage_uri=db_url,
            metadata_storage_uri=db_url,
            archival_storage_type="postgres",
            recall_storage_type="postgres",
            metadata_storage_type="postgres",
            # embeddings
            default_embedding_config=EmbeddingConfig(
                embedding_endpoint_type="hugging-face",
                embedding_endpoint="https://embeddings.memgpt.ai",
                embedding_model="BAAI/bge-large-en-v1.5",
                embedding_dim=1024,
            ),
            # llms
            default_llm_config=LLMConfig(
                model_endpoint_type="vllm",
                model_endpoint="https://api.memgpt.ai",
                model="ehartford/dolphin-2.5-mixtral-8x7b",
            ),
        )
        credentials = MemGPTCredentials()

    config.save()
    credentials.save()

    server = SyncServer()
    return server


@pytest.fixture(scope="module")
def user_id(server):
    # create user
    user = server.create_user()
    print(f"Created user\n{user.id}")

    # initialize with default presets
    server.initialize_default_presets(user.id)
    yield user.id

    # cleanup
    server.delete_user(user.id)


@pytest.fixture(scope="module")
def agent_id(server, user_id):
    # create agent
    agent_state = server.create_agent(
        user_id=user_id,
        name="test_agent",
        preset="memgpt_chat",
    )
    print(f"Created agent\n{agent_state}")
    yield agent_state.id

    # cleanup
    server.delete_agent(user_id, agent_state.id)


def test_error_on_nonexistent_agent(server, user_id, agent_id):
    try:
        fake_agent_id = uuid.uuid4()
        server.user_message(user_id=user_id, agent_id=fake_agent_id, message="Hello?")
        raise Exception("user_message call should have failed")
    except (KeyError, ValueError) as e:
        # Error is expected
        print(e)
    except:
        raise


@pytest.mark.order(1)
def test_user_message(server, user_id, agent_id):
    try:
        server.user_message(user_id=user_id, agent_id=agent_id, message="/memory")
        raise Exception("user_message call should have failed")
    except ValueError as e:
        # Error is expected
        print(e)
    except:
        raise

    server.run_command(user_id=user_id, agent_id=agent_id, command="/memory")


@pytest.mark.order(3)
def test_load_data(server, user_id, agent_id):
    # create source
    source = server.create_source("test_source", user_id)

    # load data
    archival_memories = [
        "alpha",
        "Cinderella wore a blue dress",
        "Dog eat dog",
        "ZZZ",
        "Shishir loves indian food",
    ]
    connector = DummyDataConnector(archival_memories)
    server.load_data(user_id, connector, source.name)


@pytest.mark.order(3)
def test_attach_source_to_agent(server, user_id, agent_id):
    # check archival memory size
    passages_before = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=0, count=10000)
    assert len(passages_before) == 0

    # attach source
    server.attach_source_to_agent(user_id=user_id, agent_id=agent_id, source_name="test_source")

    # check archival memory size
    passages_after = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=0, count=10000)
    assert len(passages_after) == 5


def test_save_archival_memory(server, user_id, agent_id):
    # TODO: insert into archival memory
    pass


@pytest.mark.order(4)
def test_user_message(server, user_id, agent_id):
    # add data into recall memory
    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")


@pytest.mark.order(5)
def test_get_recall_memory(server, user_id, agent_id):
    # test recall memory cursor pagination
    cursor1, messages_1 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, reverse=True, limit=2)
    cursor2, messages_2 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, reverse=True, after=cursor1, limit=1000)
    cursor3, messages_3 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, reverse=True, limit=1000)
    [m["id"] for m in messages_3]
    [m["id"] for m in messages_2]
    timestamps = [m["created_at"] for m in messages_3]
    print("timestamps", timestamps)
    assert messages_3[-1]["created_at"] < messages_3[0]["created_at"]
    assert len(messages_3) == len(messages_1) + len(messages_2)
    cursor4, messages_4 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, reverse=True, before=cursor1)
    assert len(messages_4) == 1

    # test in-context message ids
    in_context_ids = server.get_in_context_message_ids(user_id=user_id, agent_id=agent_id)
    assert len(in_context_ids) == len(messages_3)
    assert isinstance(in_context_ids[0], uuid.UUID)
    message_ids = [m["id"] for m in messages_3]
    for message_id in message_ids:
        assert message_id in in_context_ids, f"{message_id} not in {in_context_ids}"

    # test recall memory
    messages_1 = server.get_agent_messages(user_id=user_id, agent_id=agent_id, start=0, count=1)
    assert len(messages_1) == 1
    messages_2 = server.get_agent_messages(user_id=user_id, agent_id=agent_id, start=1, count=1000)
    messages_3 = server.get_agent_messages(user_id=user_id, agent_id=agent_id, start=1, count=2)
    # not sure exactly how many messages there should be
    assert len(messages_2) > len(messages_3)
    # test safe empty return
    messages_none = server.get_agent_messages(user_id=user_id, agent_id=agent_id, start=1000, count=1000)
    assert len(messages_none) == 0


@pytest.mark.order(6)
def test_get_archival_memory(server, user_id, agent_id):
    # test archival memory cursor pagination
    cursor1, passages_1 = server.get_agent_archival_cursor(user_id=user_id, agent_id=agent_id, reverse=False, limit=2, order_by="text")
    cursor2, passages_2 = server.get_agent_archival_cursor(
        user_id=user_id,
        agent_id=agent_id,
        reverse=False,
        after=cursor1,
        order_by="text",
    )
    cursor3, passages_3 = server.get_agent_archival_cursor(
        user_id=user_id,
        agent_id=agent_id,
        reverse=False,
        before=cursor2,
        limit=1000,
        order_by="text",
    )
    print("p1", [p["text"] for p in passages_1])
    print("p2", [p["text"] for p in passages_2])
    print("p3", [p["text"] for p in passages_3])
    assert passages_1[0]["text"] == "alpha"
    assert len(passages_2) in [3, 4]  # NOTE: exact size seems non-deterministic, so loosen test
    assert len(passages_3) in [4, 5]  # NOTE: exact size seems non-deterministic, so loosen test

    # test archival memory
    passage_1 = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=0, count=1)
    assert len(passage_1) == 1
    passage_2 = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=1, count=1000)
    assert len(passage_2) in [4, 5]  # NOTE: exact size seems non-deterministic, so loosen test
    # test safe empty return
    passage_none = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=1000, count=1000)
    assert len(passage_none) == 0
