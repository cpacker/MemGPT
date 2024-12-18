import time
import uuid

import pytest
from letta import create_client
from letta.schemas.letta_message import FunctionCallMessage
from letta.schemas.tool_rule import ChildToolRule, InitToolRule, TerminalToolRule
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
    t1 = client.create_or_update_tool(first_secret_word)
    t2 = client.create_or_update_tool(second_secret_word)
    t3 = client.create_or_update_tool(third_secret_word)
    t4 = client.create_or_update_tool(fourth_secret_word)
    t_err = client.create_or_update_tool(auto_error)
    tools = [t1, t2, t3, t4, t_err]

    # Make tool rules
    tool_rules = [
        InitToolRule(tool_name="first_secret_word"),
        ChildToolRule(tool_name="first_secret_word", children=["second_secret_word"]),
        ChildToolRule(tool_name="second_secret_word", children=["third_secret_word"]),
        ChildToolRule(tool_name="third_secret_word", children=["fourth_secret_word"]),
        ChildToolRule(tool_name="fourth_secret_word", children=["send_message"]),
        TerminalToolRule(tool_name="send_message"),
    ]

    # Make agent state
    agent_state = setup_agent(client, config_file, agent_uuid=agent_uuid, tool_ids=[t.id for t in tools], tool_rules=tool_rules)
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


def test_check_tool_rules_with_different_models(mock_e2b_api_key_none):
    """Test that tool rules are properly checked for different model configurations."""
    client = create_client()

    config_files = [
        "tests/configs/llm_model_configs/claude-3-sonnet-20240229.json",
        "tests/configs/llm_model_configs/openai-gpt-3.5-turbo.json",
        "tests/configs/llm_model_configs/openai-gpt-4o.json",
    ]
 
    # Create two test tools
    t1_name = "first_secret_word"
    t2_name = "second_secret_word"
    t1 = client.create_or_update_tool(first_secret_word, name=t1_name)
    t2 = client.create_or_update_tool(second_secret_word, name=t2_name)
    tool_rules = [
        InitToolRule(tool_name=t1_name),
        InitToolRule(tool_name=t2_name)
    ]
    tools = [t1, t2]

    for config_file in config_files:
        # Setup tools
        agent_uuid = str(uuid.uuid4())

        if "gpt-4o" in config_file:
            # Structured output model (should work with multiple init tools)
            agent_state = setup_agent(client, config_file, agent_uuid=agent_uuid,
                                    tool_ids=[t.id for t in tools],
                                    tool_rules=tool_rules)
            assert agent_state is not None
        else:
            # Non-structured output model (should raise error with multiple init tools)
            with pytest.raises(ValueError, match="Multiple initial tools are not supported for non-structured models"):
                setup_agent(client, config_file, agent_uuid=agent_uuid,
                            tool_ids=[t.id for t in tools],
                            tool_rules=tool_rules)
        
        # Cleanup
        cleanup(client=client, agent_uuid=agent_uuid)

    # Create tool rule with single initial tool
    t3_name = "third_secret_word"
    t3 = client.create_or_update_tool(third_secret_word, name=t3_name)
    tool_rules = [
        InitToolRule(tool_name=t3_name)
    ]
    tools = [t3]
    for config_file in config_files:
        agent_uuid = str(uuid.uuid4())

        # Structured output model (should work with single init tool)
        agent_state = setup_agent(client, config_file, agent_uuid=agent_uuid,
                                tool_ids=[t.id for t in tools],
                                tool_rules=tool_rules)
        assert agent_state is not None

        cleanup(client=client, agent_uuid=agent_uuid)


def test_claude_initial_tool_rule_enforced(mock_e2b_api_key_none):
    """Test that the initial tool rule is enforced for the first message."""
    client = create_client()

    # Create tool rules that require tool_a to be called first
    t1_name = "first_secret_word"
    t2_name = "second_secret_word"
    t1 = client.create_or_update_tool(first_secret_word, name=t1_name)
    t2 = client.create_or_update_tool(second_secret_word, name=t2_name)
    tool_rules = [
        InitToolRule(tool_name=t1_name),
        ChildToolRule(tool_name=t1_name, children=[t2_name]),
    ]
    tools = [t1, t2]

    # Make agent state
    anthropic_config_file = "tests/configs/llm_model_configs/claude-3-sonnet-20240229.json"
    for i in range(3):
        agent_uuid = str(uuid.uuid4())
        agent_state = setup_agent(client, anthropic_config_file, agent_uuid=agent_uuid, tool_ids=[t.id for t in tools], tool_rules=tool_rules)
        response = client.user_message(agent_id=agent_state.id, message="What is the second secret word?")

        assert_sanity_checks(response)
        messages = response.messages

        assert_invoked_function_call(messages, "first_secret_word")
        assert_invoked_function_call(messages, "second_secret_word")

        tool_names = [t.name for t in [t1, t2]]
        tool_names += ["send_message"]
        for m in messages:
            if isinstance(m, FunctionCallMessage):
                # Check that it's equal to the first one
                assert m.function_call.name == tool_names[0]

                # Pop out first one
                tool_names = tool_names[1:]

        print(f"Passed iteration {i}")
        cleanup(client=client, agent_uuid=agent_uuid)

        # Implement exponential backoff with initial time of 10 seconds
        if i < 2:
            backoff_time = 10 * (2 ** i)
            time.sleep(backoff_time)

@pytest.mark.timeout(60)  # Sets a 60-second timeout for the test since this could loop infinitely
def test_agent_no_structured_output_with_one_child_tool(mock_e2b_api_key_none):
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)

    send_message = client.server.tool_manager.get_tool_by_name(tool_name="send_message", actor=client.user)
    archival_memory_search = client.server.tool_manager.get_tool_by_name(tool_name="archival_memory_search", actor=client.user)
    archival_memory_insert = client.server.tool_manager.get_tool_by_name(tool_name="archival_memory_insert", actor=client.user)

    # Make tool rules
    tool_rules = [
        InitToolRule(tool_name="archival_memory_search"),
        ChildToolRule(tool_name="archival_memory_search", children=["archival_memory_insert"]),
        ChildToolRule(tool_name="archival_memory_insert", children=["send_message"]),
        TerminalToolRule(tool_name="send_message"),
    ]
    tools = [send_message, archival_memory_search, archival_memory_insert]

    config_files = [
        "tests/configs/llm_model_configs/claude-3-sonnet-20240229.json",
        "tests/configs/llm_model_configs/openai-gpt-4o.json",
    ]

    for config in config_files:
        agent_state = setup_agent(client, config, agent_uuid=agent_uuid, tool_ids=[t.id for t in tools], tool_rules=tool_rules)
        response = client.user_message(agent_id=agent_state.id, message="hi. run archival memory search")

        # Make checks
        assert_sanity_checks(response)

        # Assert the tools were called
        assert_invoked_function_call(response.messages, "archival_memory_search")
        assert_invoked_function_call(response.messages, "archival_memory_insert")
        assert_invoked_function_call(response.messages, "send_message")

        # Check ordering of tool calls
        tool_names = [t.name for t in [archival_memory_search, archival_memory_insert, send_message]]
        for m in response.messages:
            if isinstance(m, FunctionCallMessage):
                # Check that it's equal to the first one
                assert m.function_call.name == tool_names[0]

                # Pop out first one
                tool_names = tool_names[1:]

        print(f"Got successful response from client: \n\n{response}")
        cleanup(client=client, agent_uuid=agent_uuid)
