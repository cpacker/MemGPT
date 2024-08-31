import os
import threading
import time
import uuid

import pytest
from dotenv import load_dotenv

from memgpt import Admin, create_client
from memgpt.constants import DEFAULT_PRESET
from memgpt.schemas.enums import JobStatus
from memgpt.schemas.message import Message
from memgpt.schemas.usage import MemGPTUsageStatistics

# from tests.utils import create_config

test_agent_name = f"test_client_{str(uuid.uuid4())}"
# test_preset_name = "test_preset"
test_preset_name = DEFAULT_PRESET
test_agent_state = None
client = None

test_agent_state_post_message = None


# admin credentials
test_server_token = "test_server_token"


def run_server():
    load_dotenv()

    # _reset_config()

    from memgpt.server.rest_api.server import start_server

    print("Starting server...")
    start_server(debug=True)


# Fixture to create clients with different configurations
@pytest.fixture(
    # params=[{"server": True}, {"server": False}],  # whether to use REST API server
    params=[{"server": True}],  # whether to use REST API server
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
        user = admin.create_user()  # Adjust as per your client's method
        api_key = admin.create_key(user.id)
        client = create_client(base_url=server_url, token=api_key.key)  # This yields control back to the test function
    else:
        # use local client (no server)
        server_url = None
        client = create_client()

    try:
        yield client
    finally:
        # cleanup user
        if server_url:
            admin.delete_user(user.id)


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client):
    agent_state = client.create_agent(name=test_agent_name)
    print("AGENT ID", agent_state.id)
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


def test_agent(client, agent):

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
    # _reset_config()

    memory_response = client.get_in_context_memory(agent_id=agent.id)
    print("MEMORY", memory_response.compile())

    updated_memory = {"human": "Updated human memory", "persona": "Updated persona memory"}
    client.update_in_context_memory(agent_id=agent.id, section="human", value=updated_memory["human"])
    client.update_in_context_memory(agent_id=agent.id, section="persona", value=updated_memory["persona"])
    updated_memory_response = client.get_in_context_memory(agent_id=agent.id)
    assert (
        updated_memory_response.get_block("human").value == updated_memory["human"]
        and updated_memory_response.get_block("persona").value == updated_memory["persona"]
    ), "Memory update failed"


def test_agent_interactions(client, agent):
    # _reset_config()

    message = "Hello, agent!"
    print("Sending message", message)
    response = client.user_message(agent_id=agent.id, message=message)
    print("Response", response)
    assert isinstance(response.usage, MemGPTUsageStatistics)
    assert response.usage.step_count == 1
    assert response.usage.total_tokens > 0
    assert response.usage.completion_tokens > 0
    assert isinstance(response.messages[0], Message)
    print(response.messages)

    # TODO: add streaming tests


def test_archival_memory(client, agent):
    # _reset_config()

    memory_content = "Archival memory content"
    insert_response = client.insert_archival_memory(agent_id=agent.id, memory=memory_content)[0]
    print("Inserted memory", insert_response.text, insert_response.id)
    assert insert_response, "Inserting archival memory failed"

    archival_memory_response = client.get_archival_memory(agent_id=agent.id, limit=1)
    archival_memories = [memory.text for memory in archival_memory_response]
    assert memory_content in archival_memories, f"Retrieving archival memory failed: {archival_memories}"

    memory_id_to_delete = archival_memory_response[0].id
    client.delete_archival_memory(agent_id=agent.id, memory_id=memory_id_to_delete)

    # add archival memory
    memory_str = "I love chats"
    passage = client.insert_archival_memory(agent.id, memory=memory_str)[0]

    # list archival memory
    passages = client.get_archival_memory(agent.id)
    assert passage.text in [p.text for p in passages], f"Missing passage {passage.text} in {passages}"

    # get archival memory summary
    archival_summary = client.get_archival_memory_summary(agent.id)
    assert archival_summary.size == 1, f"Archival memory summary size is {archival_summary.size}"

    # delete archival memory
    client.delete_archival_memory(agent.id, passage.id)

    # TODO: check deletion
    client.get_archival_memory(agent.id)


def test_core_memory(client, agent):
    response = client.send_message(agent_id=agent.id, message="Update your core memory to remember that my name is Timber!", role="user")
    print("Response", response)

    memory = client.get_in_context_memory(agent_id=agent.id)
    assert "Timber" in memory.get_block("human").value, f"Updating core memory failed: {memory.get_block('human').value}"


def test_messages(client, agent):
    # _reset_config()

    send_message_response = client.send_message(agent_id=agent.id, message="Test message", role="user")
    assert send_message_response, "Sending message failed"

    messages_response = client.get_messages(agent_id=agent.id, limit=1)
    assert len(messages_response) > 0, "Retrieving messages failed"


def test_humans_personas(client, agent):
    # _reset_config()

    humans_response = client.list_humans()
    print("HUMANS", humans_response)

    personas_response = client.list_personas()
    print("PERSONAS", personas_response)

    persona_name = "TestPersona"
    persona_id = client.get_persona_id(persona_name)
    if persona_id:
        client.delete_persona(persona_id)
    persona = client.create_persona(name=persona_name, text="Persona text")
    assert persona.name == persona_name
    assert persona.value == "Persona text", "Creating persona failed"

    human_name = "TestHuman"
    human_id = client.get_human_id(human_name)
    if human_id:
        client.delete_human(human_id)
    human = client.create_human(name=human_name, text="Human text")
    assert human.name == human_name
    assert human.value == "Human text", "Creating human failed"


# def test_tools(client, agent):
#    tools_response = client.list_tools()
#    print("TOOLS", tools_response)
#
#    tool_name = "TestTool"
#    tool_response = client.create_tool(name=tool_name, source_code="print('Hello World')", source_type="python")
#    assert tool_response, "Creating tool failed"


def test_config(client, agent):
    # _reset_config()

    models_response = client.list_models()
    print("MODELS", models_response)

    embeddings_response = client.list_embedding_models()
    print("EMBEDDINGS", embeddings_response)

    # TODO: add back
    # config_response = client.get_config()
    # TODO: ensure config is the same as the one in the server
    # print("CONFIG", config_response)


def test_sources(client, agent):
    # _reset_config()

    # clear sources
    for source in client.list_sources():
        client.delete_source(source.id)

    # list sources
    sources = client.list_sources()
    print("listed sources", sources)
    assert len(sources) == 0

    # create a source
    source = client.create_source(name="test_source")

    # list sources
    sources = client.list_sources()
    print("listed sources", sources)
    assert len(sources) == 1

    # TODO: add back?
    assert sources[0].metadata_["num_passages"] == 0
    assert sources[0].metadata_["num_documents"] == 0

    # update the source
    original_id = source.id
    original_name = source.name
    new_name = original_name + "_new"
    client.update_source(source_id=source.id, name=new_name)

    # get the source name (check that it's been updated)
    source = client.get_source(source_id=source.id)
    assert source.name == new_name
    assert source.id == original_id

    # get the source id (make sure that it's the same)
    assert str(original_id) == client.get_source_id(source_name=new_name)

    # check agent archival memory size
    archival_memories = client.get_archival_memory(agent_id=agent.id)
    print(archival_memories)
    assert len(archival_memories) == 0

    # load a file into a source (non-blocking job)
    filename = "tests/data/memgpt_paper.pdf"
    upload_job = client.load_file_into_source(filename=filename, source_id=source.id, blocking=False)
    print("Upload job", upload_job, upload_job.status, upload_job.metadata_)

    # view active jobs
    active_jobs = client.list_active_jobs()
    jobs = client.list_jobs()
    print(jobs)
    assert upload_job.id in [j.id for j in jobs]
    assert len(active_jobs) == 1

    # wait for job to finish (with timeout)
    timeout = 60
    start_time = time.time()
    while True:
        status = client.get_job(upload_job.id).status
        print(status)
        if status == JobStatus.completed:
            break
        time.sleep(1)
        if time.time() - start_time > timeout:
            raise ValueError("Job did not finish in time")
    job = client.get_job(upload_job.id)
    created_passages = job.metadata_["num_passages"]

    # TODO: add test for blocking job

    # TODO: make sure things run in the right order
    archival_memories = client.get_archival_memory(agent_id=agent.id)
    assert len(archival_memories) == 0

    # attach a source
    client.attach_source_to_agent(source_id=source.id, agent_id=agent.id)

    # list attached sources
    attached_sources = client.list_attached_sources(agent_id=agent.id)
    print("attached sources", attached_sources)
    assert source.id in [s.id for s in attached_sources], f"Attached sources: {attached_sources}"

    # list archival memory
    archival_memories = client.get_archival_memory(agent_id=agent.id)
    # print(archival_memories)
    assert len(archival_memories) == created_passages

    # check number of passages
    sources = client.list_sources()
    # TODO: add back?
    # assert sources.sources[0].metadata_["num_passages"] > 0
    # assert sources.sources[0].metadata_["num_documents"] == 0  # TODO: fix this once document store added
    print(sources)

    # detach the source
    # TODO: add when implemented
    # client.detach_source(source.name, agent.id)

    # delete the source
    client.delete_source(source.id)
