import os
import random
import string
import threading
import time
import unittest.mock
import uuid

import pytest
import requests
import yaml
from dotenv import load_dotenv
from rich import print

from memgpt import Admin, create_client
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.cli.cli_config import add, delete, list
from memgpt.cli.cli_load import load_directory
from memgpt.constants import DEFAULT_PRESET
from memgpt.credentials import MemGPTCredentials
from memgpt.data_types import AgentState, EmbeddingConfig, LLMConfig, User
from memgpt.metadata import MetadataStore
from memgpt.settings import settings
from memgpt.utils import get_human_text, get_persona_text
from tests.config import TestMGPTConfig

from .utils import wipe_config

test_agent_name = f"test_client_{str(uuid.uuid4())}"
# test_preset_name = "test_preset"
test_preset_name = DEFAULT_PRESET
test_agent_state = None
client = None

test_agent_state_post_message = None
test_user_id = uuid.uuid4()


# admin credentials
test_server_token = "test_server_token"


@pytest.mark.skip(reason="Temporary yaml file_1 for `test_crud_preset`.")
@pytest.fixture
def create_yaml_file_1(tmp_path):
    yaml_file = tmp_path / f"test_crud_preset_file_1.yaml"
    data = {"system_prompt": "memgpt_chat", "functions": ["send_message", "pause_heartbeats"]}
    with open(yaml_file, "w") as file:
        yaml.safe_dump(data, file)
    yield yaml_file
    os.remove(yaml_file)


@pytest.mark.skip(reason="Temporary yaml file_2 for `test_crud_preset`.")
@pytest.fixture
def create_yaml_file_2(tmp_path):
    yaml_file = tmp_path / f"test_crud_preset_file_2.yaml"
    data = {"system_prompt": "memgpt_chat", "functions": ["conversation_search_date", "archival_memory_insert"]}
    with open(yaml_file, "w") as file:
        yaml.safe_dump(data, file)
    yield yaml_file
    os.remove(yaml_file)


def _reset_config():
    db_url = settings.memgpt_pg_uri

    if os.getenv("OPENAI_API_KEY"):
        config = TestMGPTConfig(
            archival_storage_uri=db_url,
            recall_storage_uri=db_url,
            metadata_storage_uri=db_url,
            archival_storage_type="chroma",
            recall_storage_type="sqlite",
            metadata_storage_type="sqlite",
            default_embedding_config=EmbeddingConfig(
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_model="text-embedding-ada-002",
                embedding_dim=1536,
            ),
            default_llm_config=LLMConfig(
                model_endpoint_type="openai",
                model_endpoint="https://api.openai.com/v1",
                model="gpt-4",
            ),
        )
        credentials = MemGPTCredentials(openai_key=os.getenv("OPENAI_API_KEY"))
    else:
        config = TestMGPTConfig(
            archival_storage_uri=db_url,
            recall_storage_uri=db_url,
            metadata_storage_uri=db_url,
            archival_storage_type="chroma",
            recall_storage_type="sqlite",
            metadata_storage_type="sqlite",
            default_embedding_config=EmbeddingConfig(
                embedding_endpoint_type="hugging-face",
                embedding_endpoint="https://embeddings.memgpt.ai",
                embedding_model="BAAI/bge-large-en-v1.5",
                embedding_dim=1024,
            ),
            default_llm_config=LLMConfig(
                model_endpoint_type="vllm",
                model_endpoint="https://api.memgpt.ai",
                model="ehartford/dolphin-2.5-mixtral-8x7b",
            ),
        )
        credentials = MemGPTCredentials()

    config.save()
    credentials.save()
    print("_reset_config :: ", config.config_path)
    return config


def run_server():
    load_dotenv()
    _reset_config()
    from memgpt.server.rest_api.server import start_server

    print("Starting server...")
    start_server(debug=True)


def wait_for_server(server_url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(server_url)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {server_url} did not become ready in {timeout} seconds.")


@pytest.fixture(scope="module")
def client(request):
    # get URL from environment
    server_url = os.getenv("MEMGPT_BASE_URL")
    token = os.getenv("MEMGPT_SERVER_PASS")
    if server_url is None:
        server_url = "http://localhost:8283"
        os.environ["MEMGPT_BASE_URL"] = server_url
    if token is None:
        token = test_server_token
        os.environ["MEMGPT_SERVER_PASS"] = test_server_token

    settings.server_pass = token

    print("Starting server thread")
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    wait_for_server(server_url)

    print("Running client tests with server:", server_url)
    # create user via admin client
    admin = Admin(server_url, token)
    response = admin.create_user(test_user_id)  # Adjust as per your client's method
    response.user_id
    token = response.api_key

    client = create_client(base_url=server_url, token=token)  # This yields control back to the test function
    try:
        yield client
    finally:
        # cleanup user
        if server_url:
            admin.delete_user(test_user_id)  # Adjust as per your client's method
        # os.environ['MEMGPT_BASE_URL'] = original_server_url
        # os.environ['MEMGPT_SERVER_PASS'] = original_token


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client):
    agent_state = client.create_agent(name=test_agent_name, preset=test_preset_name)
    print("AGENT ID", agent_state.id)
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


@pytest.mark.skip(reason="This is a helper function.")
def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choices(characters, k=length))
    return random_string


def test_crud_human(client, agent, capsys):
    # Initialize values that won't interfere with existing ones
    human_1 = generate_random_string(16)
    text_1 = generate_random_string(32)
    human_2 = generate_random_string(16)
    text_2 = generate_random_string(32)

    # Add inital human
    add("human", human_1, text_1)

    # Expect inital human to be listed
    list("humans")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(human_1) :]

    assert human_1 in output
    assert text_1 in output

    # Add second human
    add("human", human_2, text_2)

    # Expect to see second human
    list("humans")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(human_1) :]

    assert human_1 in output
    assert text_1 in output
    assert human_2 in output
    assert text_2 in output

    # Delete second human
    delete("human", human_2)

    # Expect second human to be deleted
    list("humans")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(human_1) :]

    assert human_1 in output
    assert text_1 in output
    assert human_2 not in output
    assert text_2 not in output

    # Clean up
    delete("human", human_1)


def test_crud_persona(capsys):

    # Initialize values that won't interfere with existing ones
    persona_1 = generate_random_string(16)
    text_1 = generate_random_string(32)
    persona_2 = generate_random_string(16)
    text_2 = generate_random_string(32)
    text_3 = generate_random_string(32)

    # Add inital persona
    add("persona", persona_1, text_1)

    # Expect inital persona to be listed
    list("personas")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(persona_1) :]

    assert persona_1 in output
    assert text_1 in output

    # Add second human
    add("persona", persona_2, text_2)

    # Expect to see second persona
    list("personas")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(persona_1) :]

    assert persona_1 in output
    assert text_1 in output
    assert persona_2 in output
    assert text_2 in output

    with unittest.mock.patch("questionary.confirm") as mock_confirm:
        # Enables "Overwrite" to test "update" branch of "add" logic
        mock_confirm.return_value.ask.return_value = True

        # Update second human
        add("persona", persona_2, text_3)

        # Expect to see update text
        list("personas")
        captured = capsys.readouterr()
        output = captured.out[captured.out.find(persona_1) :]

        assert persona_1 in output
        assert text_1 in output
        assert persona_2 in output
        assert output.count(persona_2) == 1
        assert text_3 in output
        assert text_2 not in output

    # Delete second persona
    delete("persona", persona_2)

    # Expect second human to be deleted
    list("personas")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(persona_1) :]

    assert persona_1 in output
    assert text_1 in output
    assert persona_2 not in output
    assert text_2 not in output

    # Clean up
    delete("persona", persona_1)


def test_crud_presets(capsys, tmp_path, create_yaml_file_1, create_yaml_file_2):

    _reset_config()

    # Initialize values that won't interfere with existing ones
    file_name_1 = generate_random_string(16)
    file_path_1 = str(tmp_path / f"test_crud_preset_file_1.yaml")
    file_name_2 = generate_random_string(16)
    file_path_2 = str(tmp_path / f"test_crud_preset_file_2.yaml")

    # Add inital preset
    add(option="preset", name=file_name_1, filename=file_path_1)

    # Expect inital preset to be listed
    list("presets")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(file_name_1) :]

    assert file_name_1 in output
    assert "send_message" in output and "pause_heartbeats" in output

    # Add second preset
    add(option="preset", name=file_name_2, filename=file_path_2)

    # Expect to see second preset
    list("presets")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(file_name_1) :]

    assert file_name_1 in output
    assert "send_message" in output and "pause_heartbeats" in output
    assert file_name_2 in output
    assert "conversation_search_date" in output and "archival_memory_insert" in output

    # Delete second preset
    delete("preset", file_name_2)

    # Expect second preset to be deleted
    list("presets")
    captured = capsys.readouterr()
    output = captured.out[captured.out.find(file_name_1) :]

    assert file_name_1 in output
    assert "send_message" in output and "pause_heartbeats" in output
    assert file_name_2 not in output
    assert "conversation_search_date" not in output and "archival_memory_insert" not in output

    # Clean up
    delete("preset", file_name_1)


def test_crud_source(client, capsys):
    config = _reset_config()
    ms = MetadataStore(config)
    user = User(id=uuid.UUID(config.anon_clientid))

    # create user and agent
    agent = AgentState(
        user_id=user.id,
        name="test_agent",
        preset=config.preset,
        persona=get_persona_text(config.persona),
        human=get_human_text(config.human),
        llm_config=config.default_llm_config,
        embedding_config=config.default_embedding_config,
    )
    ms.delete_user(user.id)
    ms.create_user(user)
    user = ms.get_user(user.id)
    print("Got user:", user, config.default_embedding_config)

    # setup storage connectors
    print("Creating storage connectors...")
    user_id = user.id
    print("User ID", user_id)
    passages_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)

    # load data
    name = "test_dataset"
    cache_dir = "CONTRIBUTING.md"

    # clear out data
    print("Resetting tables with delete_table...")
    passages_conn.delete_table()
    print("Re-creating tables...")
    passages_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
    assert passages_conn.size() == 0, f"Expected 0 records, got {passages_conn.size()}: {[vars(r) for r in passages_conn.get_all()]}"

    # TEST CLI FUNCTIONALITY BEGINS HERE
    print("Loading directory")

    # CLI FUNCTION - LOAD
    load_directory(name=name, input_files=[cache_dir], recursive=False, user_id=user_id)  # cache_dir,

    # test to see if contained in storage
    print("Querying table...")

    # CLI FUNCTION - LIST
    sources = list("sources")
    assert len(sources) == 1, f"Expected 1 source, but got {len(sources)}"
    assert sources[0].name == name, f"Expected name {name}, but got {sources[0].name}"
    print("Source", sources)

    # test to see if contained in storage
    assert (
        len(passages_conn.get_all()) == passages_conn.size()
    ), f"Expected {passages_conn.size()} passages, but got {len(passages_conn.get_all())}"
    passages = passages_conn.get_all({"data_source": name})
    print("Source", [p.data_source for p in passages])
    print("All sources", [p.data_source for p in passages_conn.get_all()])
    assert len(passages) > 0, f"Expected >0 passages, but got {len(passages)}"
    assert len(passages) == passages_conn.size(), f"Expected {passages_conn.size()} passages, but got {len(passages)}"
    assert [p.data_source == name for p in passages]
    print("Passages", passages)

    # test: listing sources
    print("Querying all...")
    # CLI FUNCTION - LIST
    sources = list("sources")
    print("All sources", [s.name for s in sources])

    # cleanup
    ms.delete_user(user.id)
    ms.delete_agent(agent.id)

    # CLI FUNCTION - LIST
    delete(option="source", name=name)

    wipe_config()
