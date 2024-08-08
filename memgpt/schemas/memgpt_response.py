from typing import List

from pydantic import BaseModel, Field

from memgpt.schemas.message import Message
from memgpt.schemas.usage import MemGPTUsageStatistics


# TODO: consider moving into own file
class MemGPTResponse(BaseModel):
    messages: List[Message] = Field(..., description="The messages returned by the agent.")
    usage: MemGPTUsageStatistics = Field(..., description="The usage statistics of the agent.")
