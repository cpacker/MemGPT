from typing import List

from pydantic import BaseModel, Field

from letta.constants import DEFAULT_MESSAGE_TOOL, DEFAULT_MESSAGE_TOOL_KWARG
from letta.schemas.message import MessageCreate


class LettaRequest(BaseModel):
    messages: List[MessageCreate] = Field(..., description="The messages to be sent to the agent.")

    # Flags to support the use of AssistantMessage message types

    assistant_message_tool_name: str = Field(
        default=DEFAULT_MESSAGE_TOOL,
        description="The name of the designated message tool.",
    )
    assistant_message_tool_kwarg: str = Field(
        default=DEFAULT_MESSAGE_TOOL_KWARG,
        description="The name of the message argument in the designated message tool.",
    )


class LettaStreamingRequest(LettaRequest):
    stream_tokens: bool = Field(
        default=False,
        description="Flag to determine if individual tokens should be streamed. Set to True for token streaming (requires stream_steps = True).",
    )
