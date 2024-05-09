from typing import Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime
from sqlmodel import Field, SQLModel

from memgpt.constants import DEFAULT_HUMAN_TEXT, DEFAULT_PERSONA_TEXT


class PresetModel(BaseModel):
    name: str = Field(..., description="The name of the preset.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the preset.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user who created the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    created_at: datetime = Field(default_factory=datetime.now, description="The unix timestamp of when the preset was created.")
    system: str = Field(..., description="The system prompt of the preset.")
    persona: str = Field(default="You are a personal assistant", description="The persona of the preset.")
    human: str = Field(default=DEFAULT_HUMAN_TEXT, description="The human of the preset.")


class HumanModel(SQLModel, table=True):
    text: str = Field(default="An unknown user", description="The human text.")
    name: str = Field(..., description="The name of the human.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the human.", primary_key=True)
    user_id: Optional[uuid.UUID] = Field(..., description="The unique identifier of the user associated with the human.")


class PersonaModel(SQLModel, table=True):
    text: str = Field(default=DEFAULT_PERSONA_TEXT, description="The persona text.")
    name: str = Field(..., description="The name of the persona.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the persona.", primary_key=True)
    user_id: Optional[uuid.UUID] = Field(..., description="The unique identifier of the user associated with the persona.")
