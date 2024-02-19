import uuid
import time
import os
import threading

from memgpt import MemGPT
from memgpt.config import MemGPTConfig
from memgpt import constants
from memgpt.data_types import LLMConfig, EmbeddingConfig, Preset
from memgpt.functions.functions import load_all_function_sets
from memgpt.prompts import gpt_system
from memgpt.constants import DEFAULT_PRESET

import memgpt.client
from memgpt.client.admin import Admin
import pytest


from .utils import wipe_config
import uuid


test_agent_name = f"test_client_{str(uuid.uuid4())}"
test_preset_name = "test_preset"
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

    os.environ["MEMGPT_SERVER_PASS"] = test_server_token

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
    response = admin.create_user(test_user_id)  # Adjust as per your client's method
    print(response)

    yield response["api_key"]  # This yields control back to the test function

    # Teardown: Delete the user after the test (or after all tests if fixture scope is module/class)
    admin.delete_user(test_user_id)  # Adjust as per your client's method


# Fixture to create clients with different configurations
@pytest.fixture(params=[{"base_url": test_base_url}, {"base_url": None}], scope="module")
def client(request, user_token):

    # use token or not
    if request.param["base_url"]:
        token = user_token
    else:
        token = None

    client = memgpt.client.client.client(**request.param, token=token)  # This yields control back to the test function
    yield client


def test_create_preset(client):

    available_functions = load_all_function_sets(merge=True)
    functions_schema = [f_dict["json_schema"] for f_name, f_dict in available_functions.items()]
    preset = Preset(
        name=test_preset_name,
        user_id=test_user_id,
        description="A preset for testing the MemGPT client",
        system=gpt_system.get_system_text(DEFAULT_PRESET),
        functions_schema=functions_schema,
    )
    client.create_preset(preset)


def test_create_agent(client):

    # ensure user exists
    if not client.server.get_user(user_id=test_user_id):
        raise ValueError("User failed to be created")

    global test_agent_state
    test_agent_state = client.create_agent(
        agent_config={
            "user_id": test_user_id,
            "name": test_agent_name,
            "preset": test_preset_name,
        }
    )
    print(f"\n\n[1] CREATED AGENT {test_agent_state.id}!!!\n\tmessages={test_agent_state.state['messages']}")
    assert test_agent_state is not None


def test_user_message(client):
    """Test that we can send a message through the client"""
    assert client is not None, "Run create_agent test first"
    print(f"\n\n[2] SENDING MESSAGE TO AGENT {test_agent_state.id}!!!\n\tmessages={test_agent_state.state['messages']}")
    response = client.user_message(agent_id=test_agent_state.id, message="Hello my name is Test, Client Test")
    assert response is not None and len(response) > 0

    global test_agent_state_post_message
    client.server.active_agents[0]["agent"].update_state()
    test_agent_state_post_message = client.server.active_agents[0]["agent"].agent_state
    print(
        f"[2] MESSAGE SEND SUCCESS!!! AGENT {test_agent_state_post_message.id}\n\tmessages={test_agent_state_post_message.state['messages']}"
    )


def test_save_load(client):
    """Test that state is being persisted correctly after an /exit

    Create a new agent, and request a message

    Then trigger
    """
    assert client is not None, "Run create_agent test first"
    assert test_agent_state is not None, "Run create_agent test first"
    assert test_agent_state_post_message is not None, "Run test_user_message test first"

    # Create a new client (not thread safe), and load the same agent
    # The agent state inside should correspond to the initial state pre-message
    if os.getenv("OPENAI_API_KEY"):
        client2 = MemGPT(quickstart="openai", user_id=test_user_id)
    else:
        client2 = MemGPT(quickstart="memgpt_hosted", user_id=test_user_id)
    print(f"\n\n[3] CREATING CLIENT2, LOADING AGENT {test_agent_state.id}!")
    client2_agent_obj = client2.server._get_or_load_agent(user_id=test_user_id, agent_id=test_agent_state.id)
    client2_agent_state = client2_agent_obj.update_state()
    print(f"[3] LOADED AGENT! AGENT {client2_agent_state.id}\n\tmessages={client2_agent_state.state['messages']}")

    # assert test_agent_state == client2_agent_state, f"{vars(test_agent_state)}\n{vars(client2_agent_state)}"
    def check_state_equivalence(state_1, state_2):
        """Helper function that checks the equivalence of two AgentState objects"""
        assert state_1.keys() == state_2.keys(), f"{state_1.keys()}\n{state_2.keys}"
        for k, v1 in state_1.items():
            v2 = state_2[k]
            if isinstance(v1, LLMConfig) or isinstance(v1, EmbeddingConfig):
                assert vars(v1) == vars(v2), f"{vars(v1)}\n{vars(v2)}"
            else:
                assert v1 == v2, f"{v1}\n{v2}"

    check_state_equivalence(vars(test_agent_state), vars(client2_agent_state))

    # Now, write out the save from the original client
    # This should persist the test message into the agent state
    client.save()

    if os.getenv("OPENAI_API_KEY"):
        client3 = MemGPT(quickstart="openai", user_id=test_user_id)
    else:
        client3 = MemGPT(quickstart="memgpt_hosted", user_id=test_user_id)
    client3_agent_obj = client3.server._get_or_load_agent(user_id=test_user_id, agent_id=test_agent_state.id)
    client3_agent_state = client3_agent_obj.update_state()

    check_state_equivalence(vars(test_agent_state_post_message), vars(client3_agent_state))


if __name__ == "__main__":
    test_create_preset()
    test_create_agent()
    test_user_message()
