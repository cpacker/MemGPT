from typing import List

from pydantic import BaseModel, Field

from memgpt.schemas.message import MessageCreate


class MemGPTRequest(BaseModel):
    messages: List[MessageCreate] = Field(..., description="The messages to be sent to the agent.")
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
        description="Set True to return the raw Message object. Set False to return the Message in the format of the MemGPT API.",
    )
