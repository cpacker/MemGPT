from memgpt import MemGPT
from memgpt import constants
from memgpt.data_types import LLMConfig, EmbeddingConfig

from .utils import wipe_config


test_agent_name = "test_client_agent"
test_agent_state = None
client = None

test_agent_state_post_message = None


def test_create_agent():
    wipe_config()
    global client
    client = MemGPT(quickstart="openai")

    global test_agent_state
    test_agent_state = client.create_agent(
        agent_config={
            "name": test_agent_name,
            "persona": constants.DEFAULT_PERSONA,
            "human": constants.DEFAULT_HUMAN,
        }
    )
    assert test_agent_state is not None


def test_user_message():
    """Test that we can send a message through the client"""
    assert client is not None, "Run create_agent test first"
    response = client.user_message(agent_id=test_agent_state.id, message="Hello my name is Test, Client Test")
    assert response is not None and len(response) > 0

    global test_agent_state_post_message
    test_agent_state_post_message = client.server.active_agents[0]["agent"].to_agent_state()


def test_save_load():
    """Test that state is being persisted correctly after an /exit

    Create a new agent, and request a message

    Then trigger
    """
    assert client is not None, "Run create_agent test first"
    assert test_agent_state is not None, "Run create_agent test first"
    assert test_agent_state_post_message is not None, "Run test_user_message test first"

    # Create a new client (not thread safe), and load the same agent
    # The agent state inside should correspond to the initial state pre-message
    client2 = MemGPT(quickstart="openai")
    client2_agent_obj = client2.server._get_or_load_agent(user_id="", agent_id=test_agent_state.id)
    client2_agent_state = client2_agent_obj.to_agent_state()

    # assert test_agent_state == client2_agent_state, f"{vars(test_agent_state)}\n{vars(client2_agent_state)}"
    def check_state_equivalence(state_1, state_2):
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

    client3 = MemGPT(quickstart="openai")
    client3_agent_obj = client3.server._get_or_load_agent(user_id="", agent_id=test_agent_state.id)
    client3_agent_state = client3_agent_obj.to_agent_state()

    check_state_equivalence(vars(test_agent_state_post_message), vars(client3_agent_state))


if __name__ == "__main__":
    test_create_agent()
    test_user_message()
