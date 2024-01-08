from memgpt import MemGPT
from memgpt import constants

from .utils import wipe_config


test_agent_name = "test_client_agent"
test_agent_id = None
client = None


def test_create_agent():
    wipe_config()
    global client
    client = MemGPT(quickstart="openai")

    agent_state = client.create_agent(
        agent_config={
            "name": test_agent_name,
            "persona": constants.DEFAULT_PERSONA,
            "human": constants.DEFAULT_HUMAN,
        }
    )
    assert agent_state is not None

    global test_agent_id
    test_agent_id = agent_state.id

    return client, agent_state


def test_user_message():
    assert client is not None, "Run create_agent test first"
    response = client.user_message(agent_id=test_agent_id, message="Hello my name is Test, Client Test")
    assert response is not None and len(response) > 0


if __name__ == "__main__":
    test_create_agent()
    test_user_message()
