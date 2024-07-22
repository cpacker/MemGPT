from typing import List
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import LLMConfigModel

class ListModelsResponse(BaseModel):
    models: List[LLMConfigModel] = Field(..., description="List of model configurations.")
