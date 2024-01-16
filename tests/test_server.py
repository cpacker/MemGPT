import uuid
import os

import memgpt.utils as utils

utils.DEBUG = True
from memgpt.config import MemGPTConfig
from memgpt.server.server import SyncServer
from memgpt.data_types import EmbeddingConfig, AgentState, LLMConfig, Message
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

    # insert messages
    agent = server._get_or_load_agent(user_id=user_id, agent_id=agent_state.id)
    agent.persistence_manager.recall_memory.storage.insert_many(
        [
            Message(user_id=user_id, agent_id=agent_state.id, role="user", text="Hello?", name="agent", model="gpt4"),
            Message(user_id=user_id, agent_id=agent_state.id, role="system", text="Hi!", name="system", model="gpt4"),
            Message(user_id=user_id, agent_id=agent_state.id, role="system", text="Hi!", name="system", model="gpt4"),
        ]
    )

    # test recall memory
    messages_1 = server.get_agent_messages(user_id=user_id, agent_id=agent_state.id, start=0, count=1)
    assert len(messages_1) == 1

    messages_2 = server.get_agent_messages(user_id=user_id, agent_id=agent_state.id, start=1, count=1000)
    print("FINAL MESSAGES", messages_2)
    assert len(messages_2) == 2


if __name__ == "__main__":
    test_server()
