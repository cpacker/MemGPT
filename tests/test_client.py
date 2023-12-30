from memgpt import MemGPT
from memgpt import constants

test_agent_id = "test_client_agent"
client = MemGPT(quickstart="memgpt_hosted")


def test_create_agent():
    agent_id = client.create_agent(
        agent_config={
            "name": test_agent_id,
            "persona": constants.DEFAULT_PERSONA,
            "human": constants.DEFAULT_HUMAN,
        }
    )
    assert agent_id is not None


def test_user_message():
    response = client.user_message(agent_id=test_agent_id, message="Hello my name is Test, Client Test")
    assert response is not None and len(response) > 0


if __name__ == "__main__":
    test_create_agent()
    test_user_message()
