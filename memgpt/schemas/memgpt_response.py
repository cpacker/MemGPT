from typing import List, Union

from pydantic import BaseModel, Field

from memgpt.schemas.enums import MessageStreamStatus
from memgpt.schemas.memgpt_message import LegacyMemGPTMessage, MemGPTMessage
from memgpt.schemas.message import Message
from memgpt.schemas.usage import MemGPTUsageStatistics

# TODO: consider moving into own file


class MemGPTResponse(BaseModel):
    # messages: List[Message] = Field(..., description="The messages returned by the agent.")
    messages: Union[List[Message], List[MemGPTMessage], List[LegacyMemGPTMessage]] = Field(
        ..., description="The messages returned by the agent."
    )
    usage: MemGPTUsageStatistics = Field(..., description="The usage statistics of the agent.")


# The streaming response is either [DONE], [DONE_STEP], [DONE], an error, or a MemGPTMessage
MemGPTStreamingResponse = Union[MemGPTMessage, MessageStreamStatus]
