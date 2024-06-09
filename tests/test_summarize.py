import os
import uuid

from memgpt import create_client

from .utils import create_config, wipe_config

# test_agent_id = "test_agent"
test_agent_name = f"test_client_{str(uuid.uuid4())}"
client = None
agent_obj = None

# TODO: these tests should include looping through LLM providers, since behavior may vary across providers
# TODO: these tests should add function calls into the summarized message sequence:W


def create_test_agent():
    """Create a test agent that we can call functions on"""
    wipe_config()
    if os.getenv("OPENAI_API_KEY"):
        create_config("openai")
    else:
        create_config("memgpt_hosted")

    global client
    client = create_client()
    agent_state = client.create_agent(
        name=test_agent_name,
    )

    global agent_obj
    agent_obj = client.server._get_or_load_agent(user_id=client.user_id, agent_id=agent_state.id)


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
        agent_id=agent_obj.agent_state.id,
        message="Hey, how's it going? What do you think about this whole shindig",
    ).messages
    assert response is not None and len(response) > 0
    print(f"test_summarize: response={response}")

    response = client.user_message(
        agent_id=agent_obj.agent_state.id,
        message="Any thoughts on the meaning of life?",
    ).messages
    assert response is not None and len(response) > 0
    print(f"test_summarize: response={response}")

    response = client.user_message(agent_id=agent_obj.agent_state.id, message="Does the number 42 ring a bell?").messages
    assert response is not None and len(response) > 0
    print(f"test_summarize: response={response}")

    response = client.user_message(
        agent_id=agent_obj.agent_state.id,
        message="Would you be surprised to learn that you're actually conversing with an AI right now?",
    ).messages
    assert response is not None and len(response) > 0
    print(f"test_summarize: response={response}")

    agent_obj.summarize_messages_inplace()
    print(f"Summarization succeeded: messages[1] = \n{agent_obj.messages[1]}")
    # response = client.run_command(agent_id=agent_obj.agent_state.id, command="summarize")
