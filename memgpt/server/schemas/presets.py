from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class ListPresetsResponse(BaseModel):
    presets: List[PresetModel] = Field(..., description="List of available presets.")


class CreatePresetsRequest(BaseModel):
    name: str = Field(..., description="The name of the preset.")
    # TODO: create should not have an id
    id: Optional[str] = Field(None, description="The unique identifier of the preset.")
    # user_id: uuid.UUID = Field(..., description="The unique identifier of the user who created the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    system: str = Field(..., description="The system prompt of the preset.")
    persona: str = Field(..., description="The persona of the preset.")
    human: str = Field(..., description="The human of the preset.")
    functions_schema: List[Dict] = Field(..., description="The functions schema of the preset.")
    persona_name: Optional[str] = Field(None, description="The name of the persona of the preset.")
    human_name: Optional[str] = Field(None, description="The name of the human of the preset.")
    system_name: Optional[str] = Field(None, description="The name of the system prompt of the preset.")


class CreatePresetResponse(BaseModel):
    ## this makes no sense - TODO return the object not a wrapper for it
    preset: PresetModel = Field(..., description="The newly created preset.")
