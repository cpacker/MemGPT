import uuid
import os

from memgpt import MemGPT
from memgpt.config import MemGPTConfig
from memgpt import constants
from memgpt.data_types import LLMConfig, EmbeddingConfig, AgentState, Passage
from memgpt.embeddings import embedding_model
from memgpt.agent_store.storage import StorageConnector, TableType
from .utils import wipe_config
import uuid


test_agent_name = f"test_client_{str(uuid.uuid4())}"
test_agent_state = None
client = None

test_agent_state_post_message = None
test_user_id = uuid.uuid4()


def generate_passages(user, agent):
    # Note: the database will filter out rows that do not correspond to agent1 and test_user by default.
    texts = ["This is a test passage", "This is another test passage", "Cinderella wept"]
    embed_model = embedding_model(agent.embedding_config)
    orig_embeddings = []
    passages = []
    for text in texts:
        embedding = embed_model.get_text_embedding(text)
        orig_embeddings.append(list(embedding))
        passages.append(
            Passage(
                user_id=user.id,
                agent_id=agent.id,
                text=text,
                embedding=embedding,
                embedding_dim=agent.embedding_config.embedding_dim,
                embedding_model=agent.embedding_config.embedding_model,
            )
        )
    return passages, orig_embeddings


def test_create_user():
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping test, missing OPENAI_API_KEY")
        return

    wipe_config()

    # create client
    client = MemGPT(quickstart="openai", user_id=test_user_id)

    # create user
    user = client.server.create_user({"id": test_user_id})

    # openai: create agent
    openai_agent = client.create_agent(
        {
            "user_id": test_user_id,
            "name": "openai_agent",
        }
    )
    assert (
        openai_agent.embedding_config.embedding_endpoint_type == "openai"
    ), f"openai_agent.embedding_config.embedding_endpoint_type={openai_agent.embedding_config.embedding_endpoint_type}"

    # openai: add passages
    passages, openai_embeddings = generate_passages(user, openai_agent)
    openai_agent_run = client.server._get_or_load_agent(user_id=user.id, agent_id=openai_agent.id)
    openai_agent_run.persistence_manager.archival_memory.storage.insert_many(passages)

    # hosted: create agent
    hosted_agent = client.create_agent(
        {
            "user_id": test_user_id,
            "name": "hosted_agent",
            "embedding_config": EmbeddingConfig(
                embedding_endpoint_type="hugging-face",
                embedding_model="BAAI/bge-large-en-v1.5",
                embedding_endpoint="https://embeddings.memgpt.ai",
                embedding_dim=1024,
            ),
        }
    )
    # check to make sure endpoint overriden
    assert (
        hosted_agent.embedding_config.embedding_endpoint_type == "hugging-face"
    ), f"hosted_agent.embedding_config.embedding_endpoint_type={hosted_agent.embedding_config.embedding_endpoint_type}"

    # hosted: add passages
    passages, hosted_embeddings = generate_passages(user, hosted_agent)
    hosted_agent_run = client.server._get_or_load_agent(user_id=user.id, agent_id=hosted_agent.id)
    hosted_agent_run.persistence_manager.archival_memory.storage.insert_many(passages)

    # test passage dimentionality
    config = MemGPTConfig.load()
    storage = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user.id)
    storage.filters = {}  # clear filters to be able to get all passages
    passages = storage.get_all()
    for passage in passages:
        if passage.agent_id == hosted_agent.id:
            assert (
                passage.embedding_dim == hosted_agent.embedding_config.embedding_dim
            ), f"passage.embedding_dim={passage.embedding_dim} != hosted_agent.embedding_config.embedding_dim={hosted_agent.embedding_config.embedding_dim}"

            # ensure was in original embeddings
            embedding = passage.embedding[: passage.embedding_dim]
            assert embedding in hosted_embeddings, f"embedding={embedding} not in hosted_embeddings={hosted_embeddings}"

            # make sure all zeros
            assert not any(
                passage.embedding[passage.embedding_dim :]
            ), f"passage.embedding[passage.embedding_dim:]={passage.embedding[passage.embedding_dim:]}"
        elif passage.agent_id == openai_agent.id:
            assert (
                passage.embedding_dim == openai_agent.embedding_config.embedding_dim
            ), f"passage.embedding_dim={passage.embedding_dim} != openai_agent.embedding_config.embedding_dim={openai_agent.embedding_config.embedding_dim}"

            # ensure was in original embeddings
            embedding = passage.embedding[: passage.embedding_dim]
            assert embedding in openai_embeddings, f"embedding={embedding} not in openai_embeddings={openai_embeddings}"

            # make sure all zeros
            assert not any(
                passage.embedding[passage.embedding_dim :]
            ), f"passage.embedding[passage.embedding_dim:]={passage.embedding[passage.embedding_dim:]}"
