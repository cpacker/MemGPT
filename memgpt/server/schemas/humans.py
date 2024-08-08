from typing import List
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import HumanModel

class ListHumansResponse(BaseModel):
    humans: List[HumanModel] = Field(..., description="List of human configurations.")


class CreateHumanRequest(BaseModel):
    text: str = Field(..., description="The human text.")
    name: str = Field(..., description="The name of the human.")
