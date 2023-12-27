from memgpt import constants
from memgpt.client.client import Client
from memgpt.config import AgentConfig
from tests.utils import configure_memgpt


agent_id = "test_client_agent"


def test_create_agent():
    # Create an AgentConfig with default persona and human txt
    agent_config = AgentConfig(
        name=agent_id,
        persona=constants.DEFAULT_PERSONA,
        human=constants.DEFAULT_HUMAN,
        preset="memgpt_chat",
        model="ehartford/dolphin-2.5-mixtral-8x7b",
        model_wrapper="chatml",
        model_endpoint_type="vllm",
        model_endpoint="https://api.memgpt.ai",
    )

    client = Client()
    agent_name = client.create_agent(agent_config=agent_config)
    assert agent_name is not None


def test_user_message():
    client = Client()
    response = client.user_message(agent_id=agent_id, message="Hello I my name is Test, Client Test")
    assert response is not None and len(response) > 0


if __name__ == "__main__":
    configure_memgpt()
    test_create_agent()
    test_user_message()
