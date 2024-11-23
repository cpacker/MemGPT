import os
import uuid

from letta import create_client
from letta.schemas.letta_message import FunctionCallMessage
from letta.schemas.tool_rule import InitToolRule, TerminalToolRule, ToolRule
from tests.helpers.endpoints_helper import (
    assert_invoked_send_message_with_keyword,
    setup_agent,
)
from tests.helpers.utils import cleanup
from tests.test_model_letta_perfomance import llm_config_dir

"""
This example shows how you can constrain tool calls in your agent.

Please note that this currently only works reliably for models with Structured Outputs (e.g. gpt-4o).

Start by downloading the dependencies.
```
poetry install --all-extras
```
"""

# Tools for this example
# Generate uuid for agent name for this example
namespace = uuid.NAMESPACE_DNS
agent_uuid = str(uuid.uuid5(namespace, "agent_tool_graph"))
config_file = os.path.join(llm_config_dir, "openai-gpt-4o.json")

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


def main():
    # 1. Set up the client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)

    # 2. Add all the tools to the client
    functions = [first_secret_word, second_secret_word, third_secret_word, fourth_secret_word, auto_error]
    tools = []
    for func in functions:
        tool = client.create_tool(func)
        tools.append(tool)
    tool_names = [t.name for t in tools[:-1]]

    # 3. Create the tool rules. It must be called in this order, or there will be an error thrown.
    tool_rules = [
        InitToolRule(tool_name="first_secret_word"),
        ToolRule(tool_name="first_secret_word", children=["second_secret_word"]),
        ToolRule(tool_name="second_secret_word", children=["third_secret_word"]),
        ToolRule(tool_name="third_secret_word", children=["fourth_secret_word"]),
        ToolRule(tool_name="fourth_secret_word", children=["send_message"]),
        TerminalToolRule(tool_name="send_message"),
    ]

    # 4. Create the agent
    agent_state = setup_agent(client, config_file, agent_uuid=agent_uuid, tools=[t.name for t in tools], tool_rules=tool_rules)

    # 5. Ask for the final secret word
    response = client.user_message(agent_id=agent_state.id, message="What is the fourth secret word?")

    # 6. Here, we thoroughly check the correctness of the response
    tool_names += ["send_message"]  # Add send message because we expect this to be called at the end
    for m in response.messages:
        if isinstance(m, FunctionCallMessage):
            # Check that it's equal to the first one
            assert m.function_call.name == tool_names[0]
            # Pop out first one
            tool_names = tool_names[1:]

    # Check final send message contains "banana"
    assert_invoked_send_message_with_keyword(response.messages, "banana")
    print(f"Got successful response from client: \n\n{response}")
    cleanup(client=client, agent_uuid=agent_uuid)


if __name__ == "__main__":
    main()
