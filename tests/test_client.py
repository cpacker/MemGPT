import os
import threading
import time
import uuid
from typing import List, Union

import pytest
from dotenv import load_dotenv

from letta import create_client
from letta.client.client import LocalClient, RESTClient
from letta.constants import DEFAULT_PRESET
from letta.schemas.agent import AgentState
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import MessageStreamStatus
from letta.schemas.letta_message import (
    AssistantMessage,
    FunctionCallMessage,
    FunctionReturn,
    InternalMonologue,
    LettaMessage,
    SystemMessage,
    UserMessage,
)
from letta.schemas.letta_response import LettaResponse, LettaStreamingResponse
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.usage import LettaUsageStatistics
from letta.services.tool_manager import ToolManager
from letta.settings import model_settings
from tests.helpers.client_helper import upload_file_using_client

# from tests.utils import create_config

test_agent_name = f"test_client_{str(uuid.uuid4())}"
# test_preset_name = "test_preset"
test_preset_name = DEFAULT_PRESET
test_agent_state = None
client = None

test_agent_state_post_message = None


def run_server():
    load_dotenv()

    # _reset_config()

    from letta.server.rest_api.app import start_server

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
        server_url = os.getenv("LETTA_SERVER_URL")
        if server_url is None:
            # run server in thread
            server_url = "http://localhost:8283"
            print("Starting server thread")
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(5)
        print("Running client tests with server:", server_url)
        # create user via admin client
        client = create_client(base_url=server_url, token=None)  # This yields control back to the test function
    else:
        # use local client (no server)
        server_url = None
        client = create_client()

    client.set_default_llm_config(LLMConfig.default_config("gpt-4"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))
    yield client


# Fixture for test agent
@pytest.fixture(scope="module")
def agent(client: Union[LocalClient, RESTClient]):
    agent_state = client.create_agent(name=test_agent_name)
    print("AGENT ID", agent_state.id)
    yield agent_state

    # delete agent
    client.delete_agent(agent_state.id)


def test_agent(client: Union[LocalClient, RESTClient], agent: AgentState):

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


def test_memory(client: Union[LocalClient, RESTClient], agent: AgentState):
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


def test_agent_interactions(client: Union[LocalClient, RESTClient], agent: AgentState):
    # _reset_config()

    message = "Hello, agent!"
    print("Sending message", message)
    response = client.user_message(agent_id=agent.id, message=message, include_full_message=True)
    # Check the types coming back
    assert all([isinstance(m, Message) for m in response.messages]), "All messages should be Message"

    print("Response", response)
    assert isinstance(response.usage, LettaUsageStatistics)
    assert response.usage.step_count == 1
    assert response.usage.total_tokens > 0
    assert response.usage.completion_tokens > 0
    assert isinstance(response.messages[0], Message)
    print(response.messages)

    # test that it also works with LettaMessage
    message = "Hello again, agent!"
    print("Sending message", message)
    response = client.user_message(agent_id=agent.id, message=message, include_full_message=False)
    assert all([isinstance(m, LettaMessage) for m in response.messages]), "All messages should be LettaMessages"

    # We should also check that the types were cast properly
    print("RESPONSE MESSAGES, client type:", type(client))
    print(response.messages)
    for letta_message in response.messages:
        assert type(letta_message) in [
            SystemMessage,
            UserMessage,
            InternalMonologue,
            FunctionCallMessage,
            FunctionReturn,
            AssistantMessage,
        ], f"Unexpected message type: {type(letta_message)}"

    # TODO: add streaming tests


def test_archival_memory(client: Union[LocalClient, RESTClient], agent: AgentState):
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


def test_core_memory(client: Union[LocalClient, RESTClient], agent: AgentState):
    response = client.send_message(agent_id=agent.id, message="Update your core memory to remember that my name is Timber!", role="user")
    print("Response", response)

    memory = client.get_in_context_memory(agent_id=agent.id)
    assert "Timber" in memory.get_block("human").value, f"Updating core memory failed: {memory.get_block('human').value}"


def test_messages(client: Union[LocalClient, RESTClient], agent: AgentState):
    # _reset_config()

    send_message_response = client.send_message(agent_id=agent.id, message="Test message", role="user")
    assert send_message_response, "Sending message failed"

    messages_response = client.get_messages(agent_id=agent.id, limit=1)
    assert len(messages_response) > 0, "Retrieving messages failed"


def test_streaming_send_message(client: Union[LocalClient, RESTClient], agent: AgentState):
    if isinstance(client, LocalClient):
        pytest.skip("Skipping test_streaming_send_message because LocalClient does not support streaming")
    assert isinstance(client, RESTClient), client

    # First, try streaming just steps

    # Next, try streaming both steps and tokens
    response = client.send_message(
        agent_id=agent.id,
        message="This is a test. Repeat after me: 'banana'",
        role="user",
        stream_steps=True,
        stream_tokens=True,
    )

    # Some manual checks to run
    # 1. Check that there were inner thoughts
    inner_thoughts_exist = False
    # 2. Check that the agent runs `send_message`
    send_message_ran = False
    # 3. Check that we get all the start/stop/end tokens we want
    #    This includes all of the MessageStreamStatus enums
    done_gen = False
    done_step = False
    done = False

    # print(response)
    assert response, "Sending message failed"
    for chunk in response:
        assert isinstance(chunk, LettaStreamingResponse)
        if isinstance(chunk, InternalMonologue) and chunk.internal_monologue and chunk.internal_monologue != "":
            inner_thoughts_exist = True
        if isinstance(chunk, FunctionCallMessage) and chunk.function_call and chunk.function_call.name == "send_message":
            send_message_ran = True
        if isinstance(chunk, MessageStreamStatus):
            if chunk == MessageStreamStatus.done:
                assert not done, "Message stream already done"
                done = True
            elif chunk == MessageStreamStatus.done_step:
                assert not done_step, "Message stream already done step"
                done_step = True
            elif chunk == MessageStreamStatus.done_generation:
                assert not done_gen, "Message stream already done generation"
                done_gen = True
        if isinstance(chunk, LettaUsageStatistics):
            # Some rough metrics for a reasonable usage pattern
            assert chunk.step_count == 1
            assert chunk.completion_tokens > 10
            assert chunk.prompt_tokens > 1000
            assert chunk.total_tokens > 1000

    assert inner_thoughts_exist, "No inner thoughts found"
    assert send_message_ran, "send_message function call not found"
    assert done, "Message stream not done"
    assert done_step, "Message stream not done step"
    assert done_gen, "Message stream not done generation"


def test_humans_personas(client: Union[LocalClient, RESTClient], agent: AgentState):
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


def test_list_tools_pagination(client: Union[LocalClient, RESTClient]):
    tools = client.list_tools()
    visited_ids = {t.id: False for t in tools}

    cursor = None
    # Choose 3 for uneven buckets (only 7 default tools)
    num_tools = 3
    # Construct a complete pagination test to see if we can return all the tools eventually
    for _ in range(0, len(tools), num_tools):
        curr_tools = client.list_tools(cursor, num_tools)
        assert len(curr_tools) <= num_tools

        for curr_tool in curr_tools:
            assert curr_tool.id in visited_ids
            visited_ids[curr_tool.id] = True

        cursor = curr_tools[-1].id

    # Assert that everything has been visited
    assert all(visited_ids.values())


def test_list_tools(client: Union[LocalClient, RESTClient]):
    tools = client.add_base_tools()
    tool_names = [t.name for t in tools]
    expected = ToolManager.BASE_TOOL_NAMES
    assert sorted(tool_names) == sorted(expected)


def test_list_files_pagination(client: Union[LocalClient, RESTClient], agent: AgentState):
    # clear sources
    for source in client.list_sources():
        client.delete_source(source.id)

    # clear jobs
    for job in client.list_jobs():
        client.delete_job(job.id)

    # create a source
    source = client.create_source(name="test_source")

    # load files into sources
    file_a = "tests/data/memgpt_paper.pdf"
    file_b = "tests/data/test.txt"
    upload_file_using_client(client, source, file_a)
    upload_file_using_client(client, source, file_b)

    # Get the first file
    files_a = client.list_files_from_source(source.id, limit=1)
    assert len(files_a) == 1
    assert files_a[0].source_id == source.id

    # Use the cursor from response_a to get the remaining file
    files_b = client.list_files_from_source(source.id, limit=1, cursor=files_a[-1].id)
    assert len(files_b) == 1
    assert files_b[0].source_id == source.id

    # Check files are different to ensure the cursor works
    assert files_a[0].file_name != files_b[0].file_name

    # Use the cursor from response_b to list files, should be empty
    files = client.list_files_from_source(source.id, limit=1, cursor=files_b[-1].id)
    assert len(files) == 0  # Should be empty


def test_delete_file_from_source(client: Union[LocalClient, RESTClient], agent: AgentState):
    # clear sources
    for source in client.list_sources():
        client.delete_source(source.id)

    # clear jobs
    for job in client.list_jobs():
        client.delete_job(job.id)

    # create a source
    source = client.create_source(name="test_source")

    # load files into sources
    file_a = "tests/data/test.txt"
    upload_file_using_client(client, source, file_a)

    # Get the first file
    files_a = client.list_files_from_source(source.id, limit=1)
    assert len(files_a) == 1
    assert files_a[0].source_id == source.id

    # Delete the file
    client.delete_file_from_source(source.id, files_a[0].id)

    # Check that no files are attached to the source
    empty_files = client.list_files_from_source(source.id, limit=1)
    assert len(empty_files) == 0


def test_load_file(client: Union[LocalClient, RESTClient], agent: AgentState):
    # _reset_config()

    # clear sources
    for source in client.list_sources():
        client.delete_source(source.id)

    # clear jobs
    for job in client.list_jobs():
        client.delete_job(job.id)

    # create a source
    source = client.create_source(name="test_source")

    # load a file into a source (non-blocking job)
    filename = "tests/data/memgpt_paper.pdf"
    upload_file_using_client(client, source, filename)

    # Get the files
    files = client.list_files_from_source(source.id)
    assert len(files) == 1  # Should be condensed to one document

    # Get the memgpt paper
    file = files[0]
    assert file.file_name == "memgpt_paper.pdf"
    assert file.source_id == source.id


def test_sources(client: Union[LocalClient, RESTClient], agent: AgentState):
    # _reset_config()

    # clear sources
    for source in client.list_sources():
        client.delete_source(source.id)

    # clear jobs
    for job in client.list_jobs():
        client.delete_job(job.id)

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
    upload_job = upload_file_using_client(client, source, filename)
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
    assert len(archival_memories) == created_passages, f"Mismatched length {len(archival_memories)} vs. {created_passages}"

    # check number of passages
    sources = client.list_sources()
    # TODO: add back?
    # assert sources.sources[0].metadata_["num_passages"] > 0
    # assert sources.sources[0].metadata_["num_documents"] == 0  # TODO: fix this once document store added
    print(sources)

    # detach the source
    assert len(client.get_archival_memory(agent_id=agent.id)) > 0, "No archival memory"
    deleted_source = client.detach_source(source_id=source.id, agent_id=agent.id)
    assert deleted_source.id == source.id
    archival_memories = client.get_archival_memory(agent_id=agent.id)
    assert len(archival_memories) == 0, f"Failed to detach source: {len(archival_memories)}"
    assert source.id not in [s.id for s in client.list_attached_sources(agent.id)]

    # delete the source
    client.delete_source(source.id)


def test_message_update(client: Union[LocalClient, RESTClient], agent: AgentState):
    """Test that we can update the details of a message"""

    # create a message
    message_response = client.send_message(agent_id=agent.id, message="Test message", role="user", include_full_message=True)
    print("Messages=", message_response)
    assert isinstance(message_response, LettaResponse)
    assert isinstance(message_response.messages[-1], Message)
    message = message_response.messages[-1]

    new_text = "This exact string would never show up in the message???"
    new_message = client.update_message(message_id=message.id, text=new_text, agent_id=agent.id)
    assert new_message.text == new_text


def test_organization(client: RESTClient):
    if isinstance(client, LocalClient):
        pytest.skip("Skipping test_organization because LocalClient does not support organizations")

    # create an organization
    org_name = "test-org"
    org = client.create_org(org_name)

    # assert the id appears
    orgs = client.list_orgs()
    assert org.id in [o.id for o in orgs]

    org = client.delete_org(org.id)
    assert org.name == org_name

    # assert the id is gone
    orgs = client.list_orgs()
    assert not (org.id in [o.id for o in orgs])


def test_list_llm_models(client: RESTClient):
    """Test that if the user's env has the right api keys set, at least one model appears in the model list"""

    def has_model_endpoint_type(models: List["LLMConfig"], target_type: str) -> bool:
        return any(model.model_endpoint_type == target_type for model in models)

    models = client.list_llm_configs()
    if model_settings.groq_api_key:
        assert has_model_endpoint_type(models, "groq")
    if model_settings.azure_api_key:
        assert has_model_endpoint_type(models, "azure")
    if model_settings.openai_api_key:
        assert has_model_endpoint_type(models, "openai")
    if model_settings.gemini_api_key:
        assert has_model_endpoint_type(models, "google_ai")
    if model_settings.anthropic_api_key:
        assert has_model_endpoint_type(models, "anthropic")
