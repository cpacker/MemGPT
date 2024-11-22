import uuid

import pytest

from letta import create_client
from letta.schemas.letta_message import FunctionCallMessage
from letta.schemas.tool_rule import InitToolRule, TerminalToolRule, ToolRule
from letta.settings import tool_settings
from tests.helpers.endpoints_helper import (
    assert_invoked_function_call,
    assert_invoked_send_message_with_keyword,
    assert_sanity_checks,
    setup_agent,
)
from tests.helpers.utils import cleanup

# Generate uuid for agent name for this example
namespace = uuid.NAMESPACE_DNS
agent_uuid = str(uuid.uuid5(namespace, "test_agent_tool_graph"))
config_file = "tests/configs/llm_model_configs/openai-gpt-4o.json"


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


"""Contrived tools for this test case"""


def first_secret_word():
    """
    Call this to retrieve the first secret word, which you will need for the second_secret_word function.
    """
    return "v0iq020i0g"


def second_secret_word(prev_secret_word: str):
    """
    Call this to retrieve the second secret word, which you will need for the third_secret_word function. If you get the word wrong, this function will error.

    Args:
        prev_secret_word (str): The secret word retrieved from calling first_secret_word.
    """
    if prev_secret_word != "v0iq020i0g":
        raise RuntimeError(f"Expected secret {"v0iq020i0g"}, got {prev_secret_word}")

    return "4rwp2b4gxq"


def third_secret_word(prev_secret_word: str):
    """
    Call this to retrieve the third secret word, which you will need for the fourth_secret_word function. If you get the word wrong, this function will error.

    Args:
        prev_secret_word (str): The secret word retrieved from calling second_secret_word.
    """
    if prev_secret_word != "4rwp2b4gxq":
        raise RuntimeError(f"Expected secret {"4rwp2b4gxq"}, got {prev_secret_word}")

    return "hj2hwibbqm"


def fourth_secret_word(prev_secret_word: str):
    """
    Call this to retrieve the last secret word, which you will need to output in a send_message later. If you get the word wrong, this function will error.

    Args:
        prev_secret_word (str): The secret word retrieved from calling third_secret_word.
    """
    if prev_secret_word != "hj2hwibbqm":
        raise RuntimeError(f"Expected secret {"hj2hwibbqm"}, got {prev_secret_word}")

    return "banana"


def auto_error():
    """
    If you call this function, it will throw an error automatically.
    """
    raise RuntimeError("This should never be called.")


@pytest.mark.timeout(60)  # Sets a 60-second timeout for the test since this could loop infinitely
def test_single_path_agent_tool_call_graph(mock_e2b_api_key_none):
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)

    # Add tools
    t1 = client.create_tool(first_secret_word)
    t2 = client.create_tool(second_secret_word)
    t3 = client.create_tool(third_secret_word)
    t4 = client.create_tool(fourth_secret_word)
    t_err = client.create_tool(auto_error)
    tools = [t1, t2, t3, t4, t_err]

    # Make tool rules
    tool_rules = [
        InitToolRule(tool_name="first_secret_word"),
        ToolRule(tool_name="first_secret_word", children=["second_secret_word"]),
        ToolRule(tool_name="second_secret_word", children=["third_secret_word"]),
        ToolRule(tool_name="third_secret_word", children=["fourth_secret_word"]),
        ToolRule(tool_name="fourth_secret_word", children=["send_message"]),
        TerminalToolRule(tool_name="send_message"),
    ]

    # Make agent state
    agent_state = setup_agent(client, config_file, agent_uuid=agent_uuid, tools=[t.name for t in tools], tool_rules=tool_rules)
    response = client.user_message(agent_id=agent_state.id, message="What is the fourth secret word?")

    # Make checks
    assert_sanity_checks(response)

    # Assert the tools were called
    assert_invoked_function_call(response.messages, "first_secret_word")
    assert_invoked_function_call(response.messages, "second_secret_word")
    assert_invoked_function_call(response.messages, "third_secret_word")
    assert_invoked_function_call(response.messages, "fourth_secret_word")

    # Check ordering of tool calls
    tool_names = [t.name for t in [t1, t2, t3, t4]]
    tool_names += ["send_message"]
    for m in response.messages:
        if isinstance(m, FunctionCallMessage):
            # Check that it's equal to the first one
            assert m.function_call.name == tool_names[0]

            # Pop out first one
            tool_names = tool_names[1:]

    # Check final send message contains "done"
    assert_invoked_send_message_with_keyword(response.messages, "banana")

    print(f"Got successful response from client: \n\n{response}")
    cleanup(client=client, agent_uuid=agent_uuid)
