from typing import List, Union

from pydantic import BaseModel, Field

from letta.constants import DEFAULT_MESSAGE_TOOL, DEFAULT_MESSAGE_TOOL_KWARG
from letta.schemas.message import Message, MessageCreate


class LettaRequest(BaseModel):
    messages: Union[List[MessageCreate], List[Message]] = Field(..., description="The messages to be sent to the agent.")
    run_async: bool = Field(default=False, description="Whether to asynchronously send the messages to the agent.")  # TODO: implement

    stream_steps: bool = Field(
        default=False, description="Flag to determine if the response should be streamed. Set to True for streaming agent steps."
    )
    stream_tokens: bool = Field(
        default=False,
        description="Flag to determine if individual tokens should be streamed. Set to True for token streaming (requires stream_steps = True).",
    )

    return_message_object: bool = Field(
        default=False,
        description="Set True to return the raw Message object. Set False to return the Message in the format of the Letta API.",
    )

    # Flags to support the use of AssistantMessage message types

    use_assistant_message: bool = Field(
        default=False,
        description="[Only applicable if return_message_object is False] If true, returns AssistantMessage objects when the agent calls a designated message tool. If false, return FunctionCallMessage objects for all tool calls.",
    )

    assistant_message_function_name: str = Field(
        default=DEFAULT_MESSAGE_TOOL,
        description="[Only applicable if use_assistant_message is True] The name of the designated message tool.",
    )
    assistant_message_function_kwarg: str = Field(
        default=DEFAULT_MESSAGE_TOOL_KWARG,
        description="[Only applicable if use_assistant_message is True] The name of the message argument in the designated message tool.",
    )
