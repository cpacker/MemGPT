from memgpt import MemGPT
from memgpt import constants
from memgpt.cli.cli import QuickstartChoice
from memgpt.config import AgentConfig

agent_id = "test_client_agent"
client = MemGPT(quickstart=QuickstartChoice.memgpt_hosted)


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

    agent_name = client.create_agent(agent_config=agent_config)
    assert agent_name is not None


def test_user_message():
    response = client.user_message(agent_id=agent_id, message="Hello my name is Test, Client Test")
    assert response is not None and len(response) > 0


if __name__ == "__main__":
    test_create_agent()
    test_user_message()
