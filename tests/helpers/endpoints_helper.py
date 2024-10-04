import json
import uuid
from typing import Callable, List, Optional, Union

from letta import LocalClient, RESTClient
from letta.config import LettaConfig
from letta.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from letta.errors import (
    InvalidFunctionCallError,
    InvalidInnerMonologueError,
    LettaError,
    MissingFunctionCallError,
    MissingInnerMonologueError,
)
from letta.llm_api.llm_api_tools import unpack_inner_thoughts_from_kwargs
from letta.local_llm.constants import INNER_THOUGHTS_KWARG
from letta.schemas.agent import AgentState
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.letta_message import (
    FunctionCallMessage,
    InternalMonologue,
    LettaMessage,
)
from letta.schemas.letta_response import LettaResponse
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.schemas.openai.chat_completion_response import Choice, FunctionCall, Message
from letta.utils import get_human_text, get_persona_text

# Generate uuid for agent name for this example
namespace = uuid.NAMESPACE_DNS
agent_uuid = str(uuid.uuid5(namespace, "test-endpoints-agent"))


# ======================================================================================================================
# Section: Test Setup
# These functions help setup the test
# ======================================================================================================================


def setup_agent(
    client: Union[LocalClient, RESTClient],
    filename: str,
    embedding_config_path: str,
    memory_human_str: str = get_human_text(DEFAULT_HUMAN),
    memory_persona_str: str = get_persona_text(DEFAULT_PERSONA),
    tools: Optional[List[str]] = None,
) -> AgentState:
    config_data = json.load(open(filename, "r"))
    llm_config = LLMConfig(**config_data)
    embedding_config = EmbeddingConfig(**json.load(open(embedding_config_path)))

    # setup config
    config = LettaConfig()
    config.default_llm_config = llm_config
    config.default_embedding_config = embedding_config
    config.save()

    memory = ChatMemory(human=memory_human_str, persona=memory_persona_str)
    agent_state = client.create_agent(name=agent_uuid, llm_config=llm_config, embedding_config=embedding_config, memory=memory, tools=tools)

    return agent_state


# ======================================================================================================================
# Section: Letta Message Assertions
# These functions are validating elements of parsed Letta Messsage
# ======================================================================================================================


def assert_sanity_checks(response: LettaResponse):
    assert response is not None
    assert response.messages is not None
    assert len(response.messages) > 0


def assert_invoked_send_message_with_keyword(messages: List[LettaMessage], keyword: str) -> None:
    # Find first instance of send_message
    target_message = None
    for message in messages:
        if isinstance(message, FunctionCallMessage) and message.function_call.name == "send_message":
            target_message = message
            break

    # No messages found with `send_messages`
    if target_message is None:
        raise LettaError("Missing send_message function call")

    send_message_function_call = target_message.function_call
    try:
        arguments = json.loads(send_message_function_call.arguments)
    except:
        raise InvalidFunctionCallError(messages=[target_message], explanation="Function call arguments could not be loaded into JSON")

    # Message field not in send_message
    if "message" not in arguments:
        raise InvalidFunctionCallError(
            messages=[target_message], explanation=f"send_message function call does not have required field `message`"
        )

    # Check that the keyword is in the message arguments
    if not keyword in arguments["message"]:
        raise InvalidFunctionCallError(messages=[target_message], explanation=f"Message argument did not contain keyword={keyword}")


def assert_invoked_function_call(messages: List[LettaMessage], function_name: str) -> None:
    for message in messages:
        if isinstance(message, FunctionCallMessage) and message.function_call.name == function_name:
            # Found it, do nothing
            return

    raise MissingFunctionCallError(
        messages=messages, explanation=f"No messages were found invoking function call with name: {function_name}"
    )


def assert_inner_monologue_is_present_and_valid(messages: List[LettaMessage]) -> None:
    for message in messages:
        if isinstance(message, InternalMonologue):
            # Found it, do nothing
            return

    raise MissingInnerMonologueError(messages=messages)


# ======================================================================================================================
# Section: Raw API Assertions
# These functions are validating elements of the (close to) raw LLM API's response
# ======================================================================================================================


def assert_contains_valid_function_call(
    message: Message,
    function_call_validator: Optional[Callable[[FunctionCall], bool]] = None,
    validation_failure_summary: Optional[str] = None,
) -> None:
    """
    Helper function to check that a message contains a valid function call.

    There is an Optional parameter `function_call_validator` that specifies a validator function.
    This function gets called on the resulting function_call to validate the function is what we expect.
    """
    if (hasattr(message, "function_call") and message.function_call is not None) and (
        hasattr(message, "tool_calls") and message.tool_calls is not None
    ):
        raise InvalidFunctionCallError(messages=[message], explanation="Both function_call and tool_calls is present in the message")
    elif hasattr(message, "function_call") and message.function_call is not None:
        function_call = message.function_call
    elif hasattr(message, "tool_calls") and message.tool_calls is not None:
        # Note: We only take the first one for now. Is this a problem? @charles
        # This seems to be standard across the repo
        function_call = message.tool_calls[0].function
    else:
        # Throw a missing function call error
        raise MissingFunctionCallError(messages=[message])

    if function_call_validator and not function_call_validator(function_call):
        raise InvalidFunctionCallError(messages=[message], explanation=validation_failure_summary)


def assert_inner_monologue_is_valid(message: Message) -> None:
    """
    Helper function to check that the inner monologue is valid.
    """
    invalid_chars = '(){}[]"'
    # Sometimes the syntax won't be correct and internal syntax will leak into message
    invalid_phrases = ["functions", "send_message"]

    monologue = message.content
    for char in invalid_chars:
        if char in monologue:
            raise InvalidInnerMonologueError(messages=[message], explanation=f"{char} is in monologue")

    for phrase in invalid_phrases:
        if phrase in monologue:
            raise InvalidInnerMonologueError(messages=[message], explanation=f"{phrase} is in monologue")


def assert_contains_correct_inner_monologue(choice: Choice, inner_thoughts_in_kwargs: bool) -> None:
    """
    Helper function to check that the inner monologue exists and is valid.
    """
    # Unpack inner thoughts out of function kwargs, and repackage into choice
    if inner_thoughts_in_kwargs:
        choice = unpack_inner_thoughts_from_kwargs(choice, INNER_THOUGHTS_KWARG)

    message = choice.message
    monologue = message.content
    if not monologue or monologue is None or monologue == "":
        raise MissingInnerMonologueError(messages=[message])

    assert_inner_monologue_is_valid(message)
