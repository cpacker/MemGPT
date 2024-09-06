from typing import List

from pydantic import BaseModel, Field

from memgpt.schemas.block import Human


class ListHumansResponse(BaseModel):
    humans: List[Human] = Field(..., description="List of human configurations.")


class CreateHumanRequest(BaseModel):
    text: str = Field(..., description="The human text.")
    name: str = Field(..., description="The name of the human.")
