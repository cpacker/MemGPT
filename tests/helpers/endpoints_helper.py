import json
from typing import Callable, Optional

from letta.config import LettaConfig
from letta.errors import (
    InvalidFunctionCallError,
    InvalidInnerMonologueError,
    MissingFunctionCallError,
    MissingInnerMonologueError,
)
from letta.llm_api.llm_api_tools import unpack_inner_thoughts_from_kwargs
from letta.local_llm.constants import INNER_THOUGHTS_KWARG
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.openai.chat_completion_response import Choice, FunctionCall, Message


def setup_llm_endpoint(filename: str, embedding_config_path: str) -> [LLMConfig, EmbeddingConfig]:
    config_data = json.load(open(filename, "r"))
    llm_config = LLMConfig(**config_data)
    embedding_config = EmbeddingConfig(**json.load(open(embedding_config_path)))

    # setup config
    config = LettaConfig()
    config.default_llm_config = llm_config
    config.default_embedding_config = embedding_config
    config.save()

    return llm_config, embedding_config


def assert_contains_valid_function_call(message: Message, function_call_validator: Optional[Callable[[FunctionCall], bool]] = None) -> None:
    """
    Helper function to check that a message contains a valid function call.

    There is an Optional parameter `function_call_validator` that specifies a validator function.
    This function gets called on the resulting function_call to validate the function is what we expect.
    """
    if (hasattr(message, "function_call") and message.function_call is not None) and (
        hasattr(message, "tool_calls") and message.tool_calls is not None
    ):
        return False
    elif hasattr(message, "function_call") and message.function_call is not None:
        function_call = message.function_call
    elif hasattr(message, "tool_calls") and message.tool_calls is not None:
        function_call = message.tool_calls[0].function
    else:
        # Throw a missing function call error
        raise MissingFunctionCallError(message=message)

    if function_call_validator and not function_call_validator(function_call):
        raise InvalidFunctionCallError(message=message)


def inner_monologue_is_valid(monologue: str) -> bool:
    invalid_chars = '(){}[]"'
    # Sometimes the syntax won't be correct and internal syntax will leak into message
    invalid_phrases = ["functions", "send_message"]

    return any(char in monologue for char in invalid_chars) or any(p in monologue for p in invalid_phrases)


def assert_contains_correct_inner_monologue(choice: Choice, inner_thoughts_in_kwargs: bool) -> None:
    if inner_thoughts_in_kwargs:
        choice = unpack_inner_thoughts_from_kwargs(choice, INNER_THOUGHTS_KWARG)

    monologue = choice.message.content
    if not monologue or monologue is None or monologue == "":
        raise MissingInnerMonologueError(message=choice.message)
    elif not inner_monologue_is_valid(monologue):
        raise InvalidInnerMonologueError(message=choice.message)
