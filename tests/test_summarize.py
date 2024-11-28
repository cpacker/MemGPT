import uuid
from typing import List

import pytest

from letta import create_client
from letta.client.client import LocalClient
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.settings import tool_settings

from .utils import wipe_config

# test_agent_id = "test_agent"
test_agent_name = f"test_client_{str(uuid.uuid4())}"
client = None
agent_obj = None

# TODO: these tests should include looping through LLM providers, since behavior may vary across providers
# TODO: these tests should add function calls into the summarized message sequence:W


@pytest.fixture
def mock_e2b_api_key_none():
    # Store the original value of e2b_api_key
    original_api_key = tool_settings.e2b_api_key

    # Set e2b_api_key to None
    tool_settings.e2b_api_key = None

    # Yield control to the test
    yield

    # Restore the original value of e2b_api_key
    tool_settings.e2b_api_key = original_api_key


def create_test_agent():
    """Create a test agent that we can call functions on"""
    wipe_config()

    global client
    client = create_client()

    client.set_default_llm_config(LLMConfig.default_config("gpt-4"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))

    agent_state = client.create_agent(
        name=test_agent_name,
    )

    global agent_obj
    agent_obj = client.server.load_agent(agent_id=agent_state.id)


def test_summarize_messages_inplace(mock_e2b_api_key_none):
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

    # reload agent object
    agent_obj = client.server.load_agent(agent_id=agent_obj.agent_state.id)

    agent_obj.summarize_messages_inplace()
    print(f"Summarization succeeded: messages[1] = \n{agent_obj.messages[1]}")
    # response = client.run_command(agent_id=agent_obj.agent_state.id, command="summarize")


def test_auto_summarize(mock_e2b_api_key_none):
    """Test that the summarizer triggers by itself"""
    client = create_client()
    client.set_default_llm_config(LLMConfig.default_config("gpt-4"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))

    small_context_llm_config = LLMConfig.default_config("gpt-4")
    # default system prompt + funcs lead to ~2300 tokens, after one message it's at 2523 tokens
    SMALL_CONTEXT_WINDOW = 4000
    small_context_llm_config.context_window = SMALL_CONTEXT_WINDOW

    agent_state = client.create_agent(
        name="small_context_agent",
        llm_config=small_context_llm_config,
    )

    try:

        def summarize_message_exists(messages: List[Message]) -> bool:
            for message in messages:
                if message.text and "The following is a summary of the previous" in message.text:
                    print(f"Summarize message found after {message_count} messages: \n {message.text}")
                    return True
            return False

        MAX_ATTEMPTS = 5
        message_count = 0
        while True:

            # send a message
            response = client.user_message(
                agent_id=agent_state.id,
                message="What is the meaning of life?",
            )
            message_count += 1

            print(f"Message {message_count}: \n\n{response.messages}" + "--------------------------------")

            # check if the summarize message is inside the messages
            assert isinstance(client, LocalClient), "Test only works with LocalClient"
            agent_obj = client.server.load_agent(agent_id=agent_state.id)
            print("SUMMARY", summarize_message_exists(agent_obj._messages))
            if summarize_message_exists(agent_obj._messages):
                break

            if message_count > MAX_ATTEMPTS:
                raise Exception(f"Summarize message not found after {message_count} messages")

    finally:
        client.delete_agent(agent_state.id)
