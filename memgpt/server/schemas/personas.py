from typing import List
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import PersonaModel

class ListPersonasResponse(BaseModel):
    personas: List[PersonaModel] = Field(..., description="List of persona configurations.")


class CreatePersonaRequest(BaseModel):
    text: str = Field(..., description="The persona text.")
    name: str = Field(..., description="The name of the persona.")
