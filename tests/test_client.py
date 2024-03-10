import uuid
import time
import threading

from memgpt import Admin, create_client
from memgpt.constants import DEFAULT_PRESET

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

    uvicorn.run(app, host="localhost", port=8283, log_level="info")


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
    user_id, token = admin.create_user(test_user_id)  # Adjust as per your client's method
    print(user_id, token)

    yield token

    # Teardown: Delete the user after the test (or after all tests if fixture scope is module/class)
    admin.delete_user(test_user_id)  # Adjust as per your client's method


# Fixture to create clients with different configurations
# @pytest.fixture(params=[{"base_url": test_base_url}, {"base_url": None}], scope="module")
@pytest.fixture(params=[{"base_url": test_base_url}], scope="module")
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

    # list sources
    sources = client.list_sources()
    print("listed sources", sources)

    # create a source
    source = client.create_source(name="test_source")

    # list sources
    sources = client.list_sources()
    print("listed sources", sources)
    assert len(sources) == 1

    # load a file into a source
    filename = "CONTRIBUTING.md"
    response = client.load_file_into_source(filename, source.id)
    print(response)

    # attach a source
    # TODO: make sure things run in the right order
    client.attach_source_to_agent(source_name="test_source", agent_id=agent.id)

    # TODO: list archival memory

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
