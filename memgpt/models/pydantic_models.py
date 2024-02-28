from typing import List, Union, Optional, Dict, Literal
from enum import Enum
from pydantic import BaseModel, Field, Json
import uuid


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


class AgentStateModel(BaseModel):
    id: uuid.UUID = Field(..., description="The unique identifier of the agent.")
    name: str = Field(..., description="The name of the agent.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the agent.")
    preset: str = Field(..., description="The preset used by the agent.")
    persona: str = Field(..., description="The persona used by the agent.")
    human: str = Field(..., description="The human used by the agent.")
    llm_config: LLMConfigModel = Field(..., description="The LLM configuration used by the agent.")
    embedding_config: EmbeddingConfigModel = Field(..., description="The embedding configuration used by the agent.")
    state: Optional[Dict] = Field(None, description="The state of the agent.")
    created_at: int = Field(..., description="The unix timestamp of when the agent was created.")
