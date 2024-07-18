import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.llm_config import LLMConfig


class AgentState(BaseModel):
    agent_id: uuid.UUID = Field(..., description="The id of the agent.")
    name: str = Field(..., description="The name of the agent.")
    description: Optional[str] = Field(None, description="The description of the agent.")

    # preset information
    tools: List[str] = Field(..., description="The tools used by the agent.")
    system: str = Field(..., description="The system prompt used by the agent.")

    # llm information
    llm_config: LLMConfig = Field(..., description="The LLM configuration used by the agent.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the agent.")

    # agent state
    state: Optional[Dict] = Field(None, description="The state of the agent.")
    metadata: Optional[Dict] = Field(None, description="The metadata of the agent.", alias="metadata_")


class CreateAgent(BaseModel):
    name: str = Field(..., description="The name of the agent.")
    description: Optional[str] = Field(None, description="The description of the agent.")
    tools: List[str] = Field(..., description="The tools used by the agent.")
    system: str = Field(..., description="The system prompt used by the agent.")
    llm_config: LLMConfig = Field(..., description="The LLM configuration used by the agent.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the agent.")
    state: Optional[Dict] = Field(None, description="The state of the agent.")
    metadata: Optional[Dict] = Field(None, description="The metadata of the agent.", alias="metadata_")


class UpdateAgentState(BaseModel):
    name: Optional[str] = Field(None, description="The name of the agent.")
    description: Optional[str] = Field(None, description="The description of the agent.")
    tools: Optional[List[str]] = Field(None, description="The tools used by the agent.")
    system: Optional[str] = Field(None, description="The system prompt used by the agent.")
    llm_config: Optional[LLMConfig] = Field(None, description="The LLM configuration used by the agent.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the agent.")
    state: Optional[Dict] = Field(None, description="The state of the agent.")
    metadata: Optional[Dict] = Field(None, description="The metadata of the agent.", alias="metadata_")
