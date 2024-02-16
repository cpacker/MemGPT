import os
import uuid

from memgpt import MemGPT
from memgpt.config import MemGPTConfig
from memgpt import constants
import memgpt.functions.function_sets.base as base_functions
from .utils import wipe_config


# test_agent_id = "test_agent"
test_agent_name = f"test_client_{str(uuid.uuid4())}"
test_user_id = uuid.uuid4()
client = None
agent_obj = None


def create_test_agent():
    """Create a test agent that we can call functions on"""
    wipe_config()
    global client
    if os.getenv("OPENAI_API_KEY"):
        client = MemGPT(quickstart="openai", user_id=test_user_id)
    else:
        client = MemGPT(quickstart="memgpt_hosted", user_id=test_user_id)

    agent_state = client.create_agent(
        agent_config={
            "user_id": test_user_id,
            "name": test_agent_name,
            "persona": constants.DEFAULT_PERSONA,
            "human": constants.DEFAULT_HUMAN,
        }
    )

    global agent_obj
    config = MemGPTConfig.load()
    agent_obj = client.server._get_or_load_agent(user_id=test_user_id, agent_id=agent_state.id)


def test_summarize():
    """Test summarization via sending the summarize CLI command or via a direct call to the agent object"""
    global client
    global agent_obj

    if agent_obj is None:
        create_test_agent()

    assert agent_obj is not None, "Run create_agent test first"
    assert client is not None, "Run create_agent test first"

    # First send a few messages (5)
    response = client.user_message(
        agent_id=agent_obj.agent_state.id, message="Hey, how's it going? What do you think about this whole shindig"
    )
    assert response is not None and len(response) > 0
    print(f"test_summarize: response={response}")

    response = client.user_message(agent_id=agent_obj.agent_state.id, message="Any thoughts on the meaning of life?")
    assert response is not None and len(response) > 0
    print(f"test_summarize: response={response}")

    response = client.user_message(agent_id=agent_obj.agent_state.id, message="Does the number 42 ring a bell?")
    assert response is not None and len(response) > 0
    print(f"test_summarize: response={response}")

    response = client.user_message(
        agent_id=agent_obj.agent_state.id, message="Would you be surprised to learn that you're actually conversing with an AI right now?"
    )
    assert response is not None and len(response) > 0
    print(f"test_summarize: response={response}")

    agent_obj.summarize_messages_inplace()
    print(f"Summarization succeeded: messages[1] = \n{agent_obj.messages[1]}")
    # response = client.run_command(agent_id=agent_obj.agent_state.id, command="summarize")
