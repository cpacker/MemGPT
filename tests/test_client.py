import uuid
import os
import time
import threading
from dotenv import load_dotenv

from memgpt import Admin, create_client
from memgpt.constants import DEFAULT_PRESET
from memgpt.data_types import Preset  # TODO move to PresetModel
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

local_service_url = "http://localhost:8283"
docker_compose_url = "http://localhost:8083"

# admin credentials
test_server_token = "test_server_token"


def run_server():
    import uvicorn
    from memgpt.server.rest_api.server import app
    from memgpt.server.rest_api.server import start_server

    load_dotenv()

    # Use os.getenv with a fallback to os.environ.get
    db_url = os.getenv("MEMGPT_PGURI") or os.environ.get("MEMGPT_PGURI")
    assert db_url, "Missing MEMGPT_PGURI"

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
                embedding_model="text-embedding-ada-002",
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

    print("Starting server...")
    start_server(debug=True)


# Fixture to create clients with different configurations
@pytest.fixture(
    params=[
        {"base_url": local_service_url},
        {"base_url": docker_compose_url},  # TODO: add when docker compose added to tests
        # {"base_url": None} # TODO: add when implemented
    ],
    scope="module",
)
# @pytest.fixture(params=[{"base_url": test_base_url}], scope="module")
def client(request):
    print("CLIENT", request.param["base_url"])
    if request.param["base_url"]:
        if request.param["base_url"] == local_service_url:
            # start server
            print("Starting server thread")
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(5)

        admin = Admin(request.param["base_url"], test_server_token)
        response = admin.create_user(test_user_id)  # Adjust as per your client's method
        user_id = response.user_id
        token = response.api_key
    else:
        token = None

    client = create_client(**request.param, token=token)  # This yields control back to the test function
    yield client

    # cleanup user
    if request.param["base_url"]:
        admin.delete_user(test_user_id)  # Adjust as per your client's method


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client):
    agent_state = client.create_agent(name=test_agent_name, preset=test_preset_name)
    print("AGENT ID", agent_state.id)
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


def test_agent(client, agent):
    # test client.rename_agent
    new_name = "RenamedTestAgent"
    client.rename_agent(agent_id=agent.id, new_name=new_name)
    renamed_agent = client.get_agent(agent_id=str(agent.id))
    assert renamed_agent.name == new_name, "Agent renaming failed"

    # test client.delete_agent and client.agent_exists
    delete_agent = client.create_agent(name="DeleteTestAgent", preset=test_preset_name)
    assert client.agent_exists(agent_id=delete_agent.id), "Agent creation failed"
    client.delete_agent(agent_id=delete_agent.id)
    assert client.agent_exists(agent_id=delete_agent.id) == False, "Agent deletion failed"


def test_memory(client, agent):
    memory_response = client.get_agent_memory(agent_id=agent.id)
    print("MEMORY", memory_response)

    updated_memory = {"human": "Updated human memory", "persona": "Updated persona memory"}
    client.update_agent_core_memory(agent_id=str(agent.id), new_memory_contents=updated_memory)
    updated_memory_response = client.get_agent_memory(agent_id=agent.id)
    assert (
        updated_memory_response.core_memory.human == updated_memory["human"]
        and updated_memory_response.core_memory.persona == updated_memory["persona"]
    ), "Memory update failed"


def test_agent_interactions(client, agent):
    message = "Hello, agent!"
    message_response = client.user_message(agent_id=str(agent.id), message=message)

    command = "/memory"
    command_response = client.run_command(agent_id=str(agent.id), command=command)
    print("command", command_response)


def test_archival_memory(client, agent):
    memory_content = "Archival memory content"
    insert_response = client.insert_archival_memory(agent_id=agent.id, memory=memory_content)
    assert insert_response, "Inserting archival memory failed"

    archival_memory_response = client.get_agent_archival_memory(agent_id=agent.id, limit=1)
    archival_memories = [memory.contents for memory in archival_memory_response.archival_memory]
    assert memory_content in archival_memories, f"Retrieving archival memory failed: {archival_memories}"

    memory_id_to_delete = archival_memory_response.archival_memory[0].id
    client.delete_archival_memory(agent_id=agent.id, memory_id=memory_id_to_delete)

    # TODO: check deletion


def test_messages(client, agent):
    send_message_response = client.send_message(agent_id=agent.id, message="Test message", role="user")
    assert send_message_response, "Sending message failed"

    messages_response = client.get_messages(agent_id=agent.id, limit=1)
    assert len(messages_response.messages) > 0, "Retrieving messages failed"


def test_humans_personas(client, agent):
    humans_response = client.list_humans()
    print("HUMANS", humans_response)

    personas_response = client.list_personas()
    print("PERSONAS", personas_response)

    persona_name = "TestPersona"
    persona = client.create_persona(name=persona_name, persona="Persona text")
    assert persona.name == persona_name
    assert persona.text == "Persona text", "Creating persona failed"

    human_name = "TestHuman"
    human = client.create_human(name=human_name, human="Human text")
    assert human.name == human_name
    assert human.text == "Human text", "Creating human failed"


def test_tools(client, agent):
    tools_response = client.list_tools()
    print("TOOLS", tools_response)

    tool_name = "TestTool"
    tool_response = client.create_tool(name=tool_name, source_code="print('Hello World')", source_type="python")
    assert tool_response, "Creating tool failed"


def test_config(client, agent):
    models_response = client.list_models()
    print("MODELS", models_response)

    # TODO: add back
    # config_response = client.get_config()
    # TODO: ensure config is the same as the one in the server
    # print("CONFIG", config_response)


def test_sources(client, agent):

    if not hasattr(client, "base_url"):
        pytest.skip("Skipping test_sources because base_url is None")

    # list sources
    sources = client.list_sources()
    print("listed sources", sources)
    assert len(sources.sources) == 0

    # create a source
    source = client.create_source(name="test_source")

    # list sources
    sources = client.list_sources()
    print("listed sources", sources)
    assert len(sources.sources) == 1
    assert sources.sources[0].metadata_["num_passages"] == 0
    assert sources.sources[0].metadata_["num_documents"] == 0

    # check agent archival memory size
    archival_memories = client.get_agent_archival_memory(agent_id=agent.id).archival_memory
    print(archival_memories)
    assert len(archival_memories) == 0

    # load a file into a source
    filename = "CONTRIBUTING.md"
    response = client.load_file_into_source(filename=filename, source_id=source.id)

    # TODO: make sure things run in the right order
    archival_memories = client.get_agent_archival_memory(agent_id=agent.id).archival_memory
    assert len(archival_memories) == 0

    # attach a source
    client.attach_source_to_agent(source_id=source.id, agent_id=agent.id)

    # list archival memory
    archival_memories = client.get_agent_archival_memory(agent_id=agent.id).archival_memory
    # print(archival_memories)
    assert len(archival_memories) == 20 or len(archival_memories) == 21

    # check number of passages
    sources = client.list_sources()
    assert sources.sources[0].metadata_["num_passages"] > 0
    assert sources.sources[0].metadata_["num_documents"] == 0  # TODO: fix this once document store added
    print(sources)

    # detach the source
    # TODO: add when implemented
    # client.detach_source(source.name, agent.id)

    # delete the source
    client.delete_source(source.id)


def test_presets(client, agent):

    new_preset = Preset(
        # user_id=client.user_id,
        name="pytest_test_preset",
        description="DUMMY_DESCRIPTION",
        system="DUMMY_SYSTEM",
        persona="DUMMY_PERSONA",
        persona_name="DUMMY_PERSONA_NAME",
        human="DUMMY_HUMAN",
        human_name="DUMMY_HUMAN_NAME",
        functions_schema=[
            {
                "name": "send_message",
                "json_schema": {
                    "name": "send_message",
                    "description": "Sends a message to the human user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Message contents. All unicode (including emojis) are supported."}
                        },
                        "required": ["message"],
                    },
                },
                "tags": ["memgpt-base"],
                "source_type": "python",
                "source_code": 'def send_message(self, message: str) -> Optional[str]:\n    """\n    Sends a message to the human user.\n\n    Args:\n        message (str): Message contents. All unicode (including emojis) are supported.\n\n    Returns:\n        Optional[str]: None is always returned as this function does not produce a response.\n    """\n    self.interface.assistant_message(message)\n    return None\n',
            }
        ],
    )

    # List all presets and make sure the preset is NOT in the list
    all_presets = client.list_presets()
    assert new_preset.id not in [p.id for p in all_presets], (new_preset, all_presets)

    # Create a preset
    client.create_preset(preset=new_preset)

    # List all presets and make sure the preset is in the list
    all_presets = client.list_presets()
    assert new_preset.id in [p.id for p in all_presets], (new_preset, all_presets)

    # Delete the preset
    client.delete_preset(preset_id=new_preset.id)

    # List all presets and make sure the preset is NOT in the list
    all_presets = client.list_presets()
    assert new_preset.id not in [p.id for p in all_presets], (new_preset, all_presets)
