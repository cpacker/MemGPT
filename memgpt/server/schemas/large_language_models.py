from typing import List

from pydantic import BaseModel, Field

from memgpt.schemas.llm_config import LLMConfig


class ListModelsResponse(BaseModel):
    models: List[LLMConfig] = Field(..., description="List of model configurations.")
