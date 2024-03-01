from typing import List, Union, Optional, Dict, Literal
from enum import Enum
from pydantic import BaseModel, Field, Json
import uuid
from datetime import datetime
from sqlmodel import Field, SQLModel

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_MEMGPT_MODEL, DEFAULT_PERSONA, DEFAULT_PRESET, LLM_MAX_TOKENS, MAX_EMBEDDING_DIM
from memgpt.utils import get_human_text, get_persona_text, printd


class LLMConfigModel(BaseModel):
    model: Optional[str] = "gpt-4"
    model_endpoint_type: Optional[str] = "openai"
    model_endpoint: Optional[str] = "https://api.openai.com/v1"
    model_wrapper: Optional[str] = None
    context_window: Optional[int] = None


class EmbeddingConfigModel(BaseModel):
    embedding_endpoint_type: Optional[str] = "openai"
    embedding_endpoint: Optional[str] = "https://api.openai.com/v1"
    embedding_model: Optional[str] = "text-embedding-ada-002"
    embedding_dim: Optional[int] = 1536
    embedding_chunk_size: Optional[int] = 300


class PresetModel(BaseModel):
    name: str = Field(..., description="The name of the preset.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the preset.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user who created the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    created_at: datetime = Field(default_factory=datetime.now, description="The unix timestamp of when the preset was created.")
    system: str = Field(..., description="The system prompt of the preset.")
    persona: str = Field(default=get_persona_text(DEFAULT_PERSONA), description="The persona of the preset.")
    human: str = Field(default=get_human_text(DEFAULT_HUMAN), description="The human of the preset.")
    functions_schema: List[Dict] = Field(..., description="The functions schema of the preset.")


class AgentStateModel(BaseModel):
    id: uuid.UUID = Field(..., description="The unique identifier of the agent.")
    name: str = Field(..., description="The name of the agent.")
    description: str = Field(None, description="The description of the agent.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the agent.")

    # timestamps
    created_at: int = Field(..., description="The unix timestamp of when the agent was created.")

    # preset information
    preset: str = Field(..., description="The preset used by the agent.")
    persona: str = Field(..., description="The persona used by the agent.")
    human: str = Field(..., description="The human used by the agent.")
    functions_schema: List[Dict] = Field(..., description="The functions schema used by the agent.")

    # llm information
    llm_config: LLMConfigModel = Field(..., description="The LLM configuration used by the agent.")
    embedding_config: EmbeddingConfigModel = Field(..., description="The embedding configuration used by the agent.")

    # agent state
    state: Optional[Dict] = Field(None, description="The state of the agent.")


class HumanModel(SQLModel, table=True):
    text: str = Field(default=get_human_text(DEFAULT_HUMAN), description="The human text.")
    name: str = Field(..., description="The name of the human.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the human.", primary_key=True)
    user_id: Optional[uuid.UUID] = Field(..., description="The unique identifier of the user associated with the human.")


class PersonaModel(SQLModel, table=True):
    text: str = Field(default=get_persona_text(DEFAULT_PERSONA), description="The persona text.")
    name: str = Field(..., description="The name of the persona.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the persona.", primary_key=True)
    user_id: Optional[uuid.UUID] = Field(..., description="The unique identifier of the user associated with the persona.")
