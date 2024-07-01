import os
import threading
import time
import uuid

import pytest
from dotenv import load_dotenv

from memgpt import Admin, create_client
from memgpt.config import MemGPTConfig
from memgpt.constants import DEFAULT_PRESET
from memgpt.credentials import MemGPTCredentials
from memgpt.data_types import Preset  # TODO move to PresetModel
from memgpt.settings import settings
from tests.utils import create_config

test_agent_name = f"test_client_{str(uuid.uuid4())}"
# test_preset_name = "test_preset"
test_preset_name = DEFAULT_PRESET
test_agent_state = None
client = None

test_agent_state_post_message = None
test_user_id = uuid.uuid4()


# admin credentials
test_server_token = "test_server_token"


def _reset_config():
    # Use os.getenv with a fallback to os.environ.get
    db_url = settings.memgpt_pg_uri

    if os.getenv("OPENAI_API_KEY"):
        create_config("openai")
        credentials = MemGPTCredentials(
            openai_key=os.getenv("OPENAI_API_KEY"),
        )
    else:  # hosted
        create_config("memgpt_hosted")
        credentials = MemGPTCredentials()

    config = MemGPTConfig.load()

    # set to use postgres
    config.archival_storage_uri = db_url
    config.recall_storage_uri = db_url
    config.metadata_storage_uri = db_url
    config.archival_storage_type = "postgres"
    config.recall_storage_type = "postgres"
    config.metadata_storage_type = "postgres"
    config.save()
    credentials.save()
    print("_reset_config :: ", config.config_path)


def run_server():
    load_dotenv()

    _reset_config()

    from memgpt.server.rest_api.server import start_server

    print("Starting server...")
    start_server(debug=True)


# Fixture to create clients with different configurations
@pytest.fixture(
    params=[{"server": True}, {"server": False}],  # whether to use REST API server
    scope="module",
)
def client(request):
    if request.param["server"]:
        # get URL from enviornment
        server_url = os.getenv("MEMGPT_SERVER_URL")
        if server_url is None:
            # run server in thread
            # NOTE: must set MEMGPT_SERVER_PASS enviornment variable
            server_url = "http://localhost:8283"
            print("Starting server thread")
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(5)
        print("Running client tests with server:", server_url)
        # create user via admin client
        admin = Admin(server_url, test_server_token)
        response = admin.create_user(test_user_id)  # Adjust as per your client's method
        token = response.api_key

    else:
        # use local client (no server)
        token = None
        server_url = None

    client = create_client(base_url=server_url, token=token)  # This yields control back to the test function
    try:
        yield client
    finally:
        # cleanup user
        if server_url:
            admin.delete_user(test_user_id)  # Adjust as per your client's method


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client):
    agent_state = client.create_agent(name=test_agent_name)
    print("AGENT ID", agent_state.id)
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


def test_agent(client, agent):
    _reset_config()

    # test client.rename_agent
    new_name = "RenamedTestAgent"
    client.rename_agent(agent_id=agent.id, new_name=new_name)
    renamed_agent = client.get_agent(agent_id=agent.id)
    assert renamed_agent.name == new_name, "Agent renaming failed"

    # test client.delete_agent and client.agent_exists
    delete_agent = client.create_agent(name="DeleteTestAgent")
    assert client.agent_exists(agent_id=delete_agent.id), "Agent creation failed"
    client.delete_agent(agent_id=delete_agent.id)
    assert client.agent_exists(agent_id=delete_agent.id) == False, "Agent deletion failed"


def test_memory(client, agent):
    _reset_config()

    memory_response = client.get_agent_memory(agent_id=agent.id)
    print("MEMORY", memory_response)

    updated_memory = {"human": "Updated human memory", "persona": "Updated persona memory"}
    client.update_agent_core_memory(agent_id=agent.id, new_memory_contents=updated_memory)
    updated_memory_response = client.get_agent_memory(agent_id=agent.id)
    assert (
        updated_memory_response.core_memory.human == updated_memory["human"]
        and updated_memory_response.core_memory.persona == updated_memory["persona"]
    ), "Memory update failed"


def test_agent_interactions(client, agent):
    _reset_config()

    message = "Hello, agent!"
    message_response = client.user_message(agent_id=agent.id, message=message)

    command = "/memory"
    command_response = client.run_command(agent_id=agent.id, command=command)
    print("command", command_response)


def test_archival_memory(client, agent):
    _reset_config()

    memory_content = "Archival memory content"
    insert_response = client.insert_archival_memory(agent_id=agent.id, memory=memory_content)
    assert insert_response, "Inserting archival memory failed"

    archival_memory_response = client.get_agent_archival_memory(agent_id=agent.id, limit=1)
    print("MEMORY")
    archival_memories = [memory.contents for memory in archival_memory_response.archival_memory]
    assert memory_content in archival_memories, f"Retrieving archival memory failed: {archival_memories}"

    memory_id_to_delete = archival_memory_response.archival_memory[0].id
    client.delete_archival_memory(agent_id=agent.id, memory_id=memory_id_to_delete)

    # TODO: check deletion


def test_messages(client, agent):
    _reset_config()

    send_message_response = client.send_message(agent_id=agent.id, message="Test message", role="user")
    assert send_message_response, "Sending message failed"

    messages_response = client.get_messages(agent_id=agent.id, limit=1)
    assert len(messages_response.messages) > 0, "Retrieving messages failed"


def test_humans_personas(client, agent):
    _reset_config()

    humans_response = client.list_humans()
    print("HUMANS", humans_response)

    personas_response = client.list_personas()
    print("PERSONAS", personas_response)

    persona_name = "TestPersona"
    if client.get_persona(persona_name):
        client.delete_persona(persona_name)
    persona = client.create_persona(name=persona_name, persona="Persona text")
    assert persona.name == persona_name
    assert persona.text == "Persona text", "Creating persona failed"

    human_name = "TestHuman"
    if client.get_human(human_name):
        client.delete_human(human_name)
    human = client.create_human(name=human_name, human="Human text")
    assert human.name == human_name
    assert human.text == "Human text", "Creating human failed"


# def test_tools(client, agent):
#    tools_response = client.list_tools()
#    print("TOOLS", tools_response)
#
#    tool_name = "TestTool"
#    tool_response = client.create_tool(name=tool_name, source_code="print('Hello World')", source_type="python")
#    assert tool_response, "Creating tool failed"


def test_config(client, agent):
    _reset_config()

    models_response = client.list_models()
    print("MODELS", models_response)

    # TODO: add back
    # config_response = client.get_config()
    # TODO: ensure config is the same as the one in the server
    # print("CONFIG", config_response)


def test_sources(client, agent):
    _reset_config()

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
    upload_job = client.load_file_into_source(filename=filename, source_id=source.id)
    print("Upload job", upload_job, upload_job.status, upload_job.metadata)

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


# def test_presets(client, agent):
#    _reset_config()
#
#    # new_preset = Preset(
#    #    # user_id=client.user_id,
#    #    name="pytest_test_preset",
#    #    description="DUMMY_DESCRIPTION",
#    #    system="DUMMY_SYSTEM",
#    #    persona="DUMMY_PERSONA",
#    #    persona_name="DUMMY_PERSONA_NAME",
#    #    human="DUMMY_HUMAN",
#    #    human_name="DUMMY_HUMAN_NAME",
#    #    functions_schema=[
#    #        {
#    #            "name": "send_message",
#    #            "json_schema": {
#    #                "name": "send_message",
#    #                "description": "Sends a message to the human user.",
#    #                "parameters": {
#    #                    "type": "object",
#    #                    "properties": {
#    #                        "message": {"type": "string", "description": "Message contents. All unicode (including emojis) are supported."}
#    #                    },
#    #                    "required": ["message"],
#    #                },
#    #            },
#    #            "tags": ["memgpt-base"],
#    #            "source_type": "python",
#    #            "source_code": 'def send_message(self, message: str) -> Optional[str]:\n    """\n    Sends a message to the human user.\n\n    Args:\n        message (str): Message contents. All unicode (including emojis) are supported.\n\n    Returns:\n        Optional[str]: None is always returned as this function does not produce a response.\n    """\n    self.interface.assistant_message(message)\n    return None\n',
#    #        }
#    #    ],
#    # )
#
#    ## List all presets and make sure the preset is NOT in the list
#    # all_presets = client.list_presets()
#    # assert new_preset.id not in [p.id for p in all_presets], (new_preset, all_presets)
#    # Create a preset
#    new_preset = client.create_preset(name="pytest_test_preset")
#
#    # List all presets and make sure the preset is in the list
#    all_presets = client.list_presets()
#    assert new_preset.id in [p.id for p in all_presets], (new_preset, all_presets)
#
#    # Delete the preset
#    client.delete_preset(preset_id=new_preset.id)
#
#    # List all presets and make sure the preset is NOT in the list
#    all_presets = client.list_presets()
#    assert new_preset.id not in [p.id for p in all_presets], (new_preset, all_presets)


# def test_tools(client, agent):
#
#    # load a function
#    file_path = "tests/data/functions/dump_json.py"
#    module_name = "dump_json"
#
#    # list functions
#    response = client.list_tools()
#    orig_tools = response.tools
#    print(orig_tools)
#
#    # add the tool
#    create_tool_response = client.create_tool(name=module_name, file_path=file_path)
#    print(create_tool_response)
#
#    # list functions
#    response = client.list_tools()
#    new_tools = response.tools
#    assert module_name in [tool.name for tool in new_tools]
#    # assert len(new_tools) == len(orig_tools) + 1
#
#    # TODO: add a function to a preset
#
#    # TODO: add a function to an agent
