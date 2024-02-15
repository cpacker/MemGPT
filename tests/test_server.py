import uuid
import os
import memgpt.utils as utils
from dotenv import load_dotenv

utils.DEBUG = True
from memgpt.config import MemGPTConfig
from memgpt.credentials import MemGPTCredentials
from memgpt.server.server import SyncServer
from memgpt.data_types import EmbeddingConfig, AgentState, LLMConfig, Message, Passage, User
from memgpt.embeddings import embedding_model
from memgpt.presets.presets import add_default_presets
from .utils import wipe_config, wipe_memgpt_home


def test_server():
    load_dotenv()
    wipe_memgpt_home()

    # Use os.getenv with a fallback to os.environ.get
    db_url = os.getenv("PGVECTOR_TEST_DB_URL") or os.environ.get("PGVECTOR_TEST_DB_URL")

    if os.getenv("OPENAI_API_KEY"):
        config = MemGPTConfig(
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
        config = MemGPTConfig(
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

    # create user
    user = server.create_user()
    print(f"Created user\n{user.id}")

    try:
        fake_agent_id = uuid.uuid4()
        server.user_message(user_id=user.id, agent_id=fake_agent_id, message="Hello?")
        raise Exception("user_message call should have failed")
    except (KeyError, ValueError) as e:
        # Error is expected
        print(e)
    except:
        raise

    # create presets
    add_default_presets(user.id, server.ms)

    # create agent
    agent_state = server.create_agent(
        user_id=user.id,
        agent_config=dict(name="test_agent", user_id=user.id, preset="memgpt_chat", human="cs_phd", persona="sam_pov"),
    )
    print(f"Created agent\n{agent_state}")

    try:
        server.user_message(user_id=user.id, agent_id=agent_state.id, message="/memory")
        raise Exception("user_message call should have failed")
    except ValueError as e:
        # Error is expected
        print(e)
    except:
        raise

    print(server.run_command(user_id=user.id, agent_id=agent_state.id, command="/memory"))

    # add data into archival memory
    agent = server._load_agent(user_id=user.id, agent_id=agent_state.id)
    archival_memories = ["alpha", "Cinderella wore a blue dress", "Dog eat dog", "ZZZ", "Shishir loves indian food"]
    embed_model = embedding_model(agent.agent_state.embedding_config)
    for text in archival_memories:
        embedding = embed_model.get_text_embedding(text)
        agent.persistence_manager.archival_memory.storage.insert(
            Passage(
                user_id=user.id,
                agent_id=agent_state.id,
                text=text,
                embedding=embedding,
                embedding_dim=agent.agent_state.embedding_config.embedding_dim,
                embedding_model=agent.agent_state.embedding_config.embedding_model,
            )
        )

    # add data into recall memory
    server.user_message(user_id=user.id, agent_id=agent_state.id, message="Hello?")
    server.user_message(user_id=user.id, agent_id=agent_state.id, message="Hello?")
    server.user_message(user_id=user.id, agent_id=agent_state.id, message="Hello?")
    server.user_message(user_id=user.id, agent_id=agent_state.id, message="Hello?")
    server.user_message(user_id=user.id, agent_id=agent_state.id, message="Hello?")

    # test recall memory cursor pagination
    cursor1, messages_1 = server.get_agent_recall_cursor(user_id=user.id, agent_id=agent_state.id, reverse=True, limit=2)
    cursor2, messages_2 = server.get_agent_recall_cursor(user_id=user.id, agent_id=agent_state.id, reverse=True, after=cursor1, limit=1000)
    cursor3, messages_3 = server.get_agent_recall_cursor(user_id=user.id, agent_id=agent_state.id, reverse=True, limit=1000)
    ids3 = [m["id"] for m in messages_3]
    ids2 = [m["id"] for m in messages_2]
    timestamps = [m["created_at"] for m in messages_3]
    print("timestamps", timestamps)
    assert messages_3[-1]["created_at"] < messages_3[0]["created_at"]
    assert len(messages_3) == len(messages_1) + len(messages_2)
    cursor4, messages_4 = server.get_agent_recall_cursor(user_id=user.id, agent_id=agent_state.id, reverse=True, before=cursor1)
    assert len(messages_4) == 1

    # test in-context message ids
    in_context_ids = server.get_in_context_message_ids(user_id=user.id, agent_id=agent_state.id)
    assert len(in_context_ids) == len(messages_3)
    assert isinstance(in_context_ids[0], uuid.UUID)
    message_ids = [m["id"] for m in messages_3]
    for message_id in message_ids:
        assert message_id in in_context_ids, f"{message_id} not in {in_context_ids}"

    # test archival memory cursor pagination
    cursor1, passages_1 = server.get_agent_archival_cursor(
        user_id=user.id, agent_id=agent_state.id, reverse=False, limit=2, order_by="text"
    )
    cursor2, passages_2 = server.get_agent_archival_cursor(
        user_id=user.id, agent_id=agent_state.id, reverse=False, after=cursor1, order_by="text"
    )
    cursor3, passages_3 = server.get_agent_archival_cursor(
        user_id=user.id, agent_id=agent_state.id, reverse=False, before=cursor2, limit=1000, order_by="text"
    )
    print("p1", [p["text"] for p in passages_1])
    print("p2", [p["text"] for p in passages_2])
    print("p3", [p["text"] for p in passages_3])
    assert passages_1[0]["text"] == "alpha"
    assert len(passages_2) == 3
    assert len(passages_3) == 4

    # test recall memory
    messages_1 = server.get_agent_messages(user_id=user.id, agent_id=agent_state.id, start=0, count=1)
    assert len(messages_1) == 1
    messages_2 = server.get_agent_messages(user_id=user.id, agent_id=agent_state.id, start=1, count=1000)
    messages_3 = server.get_agent_messages(user_id=user.id, agent_id=agent_state.id, start=1, count=5)
    # not sure exactly how many messages there should be
    assert len(messages_2) > len(messages_3)
    # test safe empty return
    messages_none = server.get_agent_messages(user_id=user.id, agent_id=agent_state.id, start=1000, count=1000)
    assert len(messages_none) == 0

    # test archival memory
    passage_1 = server.get_agent_archival(user_id=user.id, agent_id=agent_state.id, start=0, count=1)
    assert len(passage_1) == 1
    passage_2 = server.get_agent_archival(user_id=user.id, agent_id=agent_state.id, start=1, count=1000)
    assert len(passage_2) == 4
    # test safe empty return
    passage_none = server.get_agent_archival(user_id=user.id, agent_id=agent_state.id, start=1000, count=1000)
    assert len(passage_none) == 0


if __name__ == "__main__":
    test_server()
