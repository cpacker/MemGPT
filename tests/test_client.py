import uuid
import os
import time
import threading
from dotenv import load_dotenv

from memgpt.server.rest_api.server import start_server
from memgpt import Admin, create_client
from memgpt.constants import DEFAULT_PRESET
from dotenv import load_dotenv

from tests.config import TestMGPTConfig

from memgpt.credentials import MemGPTCredentials
from memgpt.data_types import EmbeddingConfig, LLMConfig
from .utils import wipe_config, wipe_memgpt_home


import pytest
import uuid

test_agent_name = f"test_client_{str(uuid.uuid4())}"
# test_preset_name = "test_preset"
test_preset_name = DEFAULT_PRESET
test_agent_state = None
client = None

test_agent_state_post_message = None
test_user_id = uuid.uuid4()

test_base_url = "http://localhost:8283"

# admin credentials
test_server_token = "test_server_token"


def run_server():
    import uvicorn
    from memgpt.server.rest_api.server import app

    load_dotenv()

    # Use os.getenv with a fallback to os.environ.get
    db_url = os.getenv("PGVECTOR_TEST_DB_URL") or os.environ.get("PGVECTOR_TEST_DB_URL")
    assert db_url, "Missing PGVECTOR_TEST_DB_URL"

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

    start_server(debug=False)


@pytest.fixture(scope="session", autouse=True)
def start_uvicorn_server():
    """Starts Uvicorn server in a background thread."""

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print("Starting server...")
    time.sleep(5)
    yield


@pytest.fixture(scope="module")
def user_token():
    # Setup: Create a user via the client before the tests

    admin = Admin(test_base_url, test_server_token)
    response = admin.create_user(test_user_id)  # Adjust as per your client's method
    user_id = response.user_id
    token = response.api_key

    yield token

    # Teardown: Delete the user after the test (or after all tests if fixture scope is module/class)
    admin.delete_user(test_user_id)  # Adjust as per your client's method


# Fixture to create clients with different configurations
@pytest.fixture(params=[{"base_url": test_base_url}, {"base_url": None}], scope="module")
# @pytest.fixture(params=[{"base_url": test_base_url}], scope="module")
def client(request, user_token):
    # use token or not
    if request.param["base_url"]:
        token = user_token
    else:
        token = None

    client = create_client(**request.param, token=token)  # This yields control back to the test function
    yield client


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client):
    agent_state = client.create_agent(name=test_agent_name, preset=test_preset_name)
    print("AGENT ID", agent_state.id)
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


# TODO: add back once REST API supports
# def test_create_preset(client):
#
#    available_functions = load_all_function_sets(merge=True)
#    functions_schema = [f_dict["json_schema"] for f_name, f_dict in available_functions.items()]
#    preset = Preset(
#        name=test_preset_name,
#        user_id=test_user_id,
#        description="A preset for testing the MemGPT client",
#        system=gpt_system.get_system_text(DEFAULT_PRESET),
#        functions_schema=functions_schema,
#    )
#    client.create_preset(preset)


# def test_create_agent(client):
#    global test_agent_state
#    test_agent_state = client.create_agent(
#        name=test_agent_name,
#        preset=test_preset_name,
#    )
#    print(f"\n\n[1] CREATED AGENT {test_agent_state.id}!!!\n\tmessages={test_agent_state.state['messages']}")
#    assert test_agent_state is not None


def test_sources(client, agent):

    if not hasattr(client, "base_url"):
        pytest.skip("Skipping test_sources because base_url is None")

    # list sources
    sources = client.list_sources()
    print("listed sources", sources)

    # create a source
    source = client.create_source(name="test_source")

    # list sources
    sources = client.list_sources()
    print("listed sources", sources)
    assert len(sources) == 1

    # check agent archival memory size
    archival_memories = client.get_agent_archival_memory(agent_id=agent.id)
    print(archival_memories)
    assert len(archival_memories) == 0

    # load a file into a source
    filename = "CONTRIBUTING.md"
    num_passages = 20
    response = client.load_file_into_source(filename, source.id)
    print(response)

    # attach a source
    # TODO: make sure things run in the right order
    client.attach_source_to_agent(source_name="test_source", agent_id=agent.id)

    # list archival memory
    archival_memories = client.get_agent_archival_memory(agent_id=agent.id)
    print(archival_memories)
    assert len(archival_memories) == num_passages

    # detach the source
    # TODO: add when implemented
    # client.detach_source(source.name, agent.id)

    # delete the source
    client.delete_source(source.id)


# def test_user_message(client, agent):
#    """Test that we can send a message through the client"""
#    assert client is not None, "Run create_agent test first"
#    print(f"\n\n[2] SENDING MESSAGE TO AGENT {agent.id}!!!\n\tmessages={agent.state['messages']}")
#    response = client.user_message(agent_id=agent.id, message="Hello my name is Test, Client Test")
#    assert response is not None and len(response) > 0
