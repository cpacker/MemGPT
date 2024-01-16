import uuid
import os

import memgpt.utils as utils

utils.DEBUG = True
from memgpt.config import MemGPTConfig
from memgpt.server.server import SyncServer
from memgpt.data_types import EmbeddingConfig, AgentState, LLMConfig, Message, Passage
from memgpt.embeddings import embedding_model
from .utils import wipe_config, wipe_memgpt_home


def test_server():
    wipe_memgpt_home()

    config = MemGPTConfig.load()
    user_id = uuid.UUID(config.anon_clientid)
    server = SyncServer()

    try:
        fake_agent_id = uuid.uuid4()
        server.user_message(user_id=user_id, agent_id=fake_agent_id, message="Hello?")
        raise Exception("user_message call should have failed")
    except (KeyError, ValueError) as e:
        # Error is expected
        print(e)
    except:
        raise

    # embedding config
    if os.getenv("OPENAI_API_KEY"):
        embedding_config = EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_dim=1536,
            openai_key=os.getenv("OPENAI_API_KEY"),
        )

    else:
        embedding_config = EmbeddingConfig(embedding_endpoint_type="local", embedding_endpoint=None, embedding_dim=384)

    agent_state = server.create_agent(
        user_id=user_id,
        agent_config=dict(
            name="test_agent", user_id=user_id, preset="memgpt_chat", human="cs_phd", persona="sam_pov", embedding_config=embedding_config
        ),
    )
    print(f"Created agent\n{agent_state}")

    try:
        server.user_message(user_id=user_id, agent_id=agent_state.id, message="/memory")
        raise Exception("user_message call should have failed")
    except ValueError as e:
        # Error is expected
        print(e)
    except:
        raise

    print(server.run_command(user_id=user_id, agent_id=agent_state.id, command="/memory"))

    server.user_message(user_id=user_id, agent_id=agent_state.id, message="Hello?")
    server.user_message(user_id=user_id, agent_id=agent_state.id, message="Hello?")
    server.user_message(user_id=user_id, agent_id=agent_state.id, message="Hello?")
    server.user_message(user_id=user_id, agent_id=agent_state.id, message="Hello?")
    server.user_message(user_id=user_id, agent_id=agent_state.id, message="Hello?")

    # test recall memory
    messages_1 = server.get_agent_messages(user_id=user_id, agent_id=agent_state.id, start=0, count=1)
    assert len(messages_1) == 1

    messages_2 = server.get_agent_messages(user_id=user_id, agent_id=agent_state.id, start=1, count=1000)
    messages_3 = server.get_agent_messages(user_id=user_id, agent_id=agent_state.id, start=1, count=5)
    # not sure exactly how many messages there should be
    assert len(messages_2) > len(messages_3)

    # test safe empty return
    messages_none = server.get_agent_messages(user_id=user_id, agent_id=agent_state.id, start=1000, count=1000)
    assert len(messages_none) == 0

    # test archival memory
    agent = server._load_agent(user_id=user_id, agent_id=agent_state.id)
    archival_memories = ["Cinderella wore a blue dress", "Dog eat dog", "Shishir loves indian food"]
    embed_model = embedding_model(embedding_config)
    for text in archival_memories:
        embedding = embed_model.get_text_embedding(text)
        agent.persistence_manager.archival_memory.storage.insert(
            Passage(user_id=user_id, agent_id=agent_state.id, text=text, embedding=embedding)
        )
    passage_1 = server.get_agent_archival(user_id=user_id, agent_id=agent_state.id, start=0, count=1)
    assert len(passage_1) == 1
    passage_2 = server.get_agent_archival(user_id=user_id, agent_id=agent_state.id, start=1, count=1000)
    assert len(passage_2) == 2

    print(passage_1)

    # test safe empty return
    passage_none = server.get_agent_archival(user_id=user_id, agent_id=agent_state.id, start=1000, count=1000)
    assert len(passage_none) == 0


if __name__ == "__main__":
    test_server()
