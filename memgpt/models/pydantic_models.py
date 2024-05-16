import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, Column
from sqlalchemy_utils import ChoiceType
from sqlmodel import Field, SQLModel

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from memgpt.utils import get_human_text, get_persona_text, get_utc_time


class LLMConfigModel(BaseModel):
    model: Optional[str] = "gpt-4"
    model_endpoint_type: Optional[str] = "openai"
    model_endpoint: Optional[str] = "https://api.openai.com/v1"
    model_wrapper: Optional[str] = None
    context_window: Optional[int] = None

    # FIXME hack to silence pydantic protected namespace warning
    model_config = ConfigDict(protected_namespaces=())


class EmbeddingConfigModel(BaseModel):
    embedding_endpoint_type: Optional[str] = "openai"
    embedding_endpoint: Optional[str] = "https://api.openai.com/v1"
    embedding_model: Optional[str] = "text-embedding-ada-002"
    embedding_dim: Optional[int] = 1536
    embedding_chunk_size: Optional[int] = 300


class PresetModel(BaseModel):
    name: str = Field(..., description="The name of the preset.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the preset.")
    user_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the user who created the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the preset was created.")
    system: str = Field(..., description="The system prompt of the preset.")
    system_name: Optional[str] = Field(None, description="The name of the system prompt of the preset.")
    persona: str = Field(default=get_persona_text(DEFAULT_PERSONA), description="The persona of the preset.")
    persona_name: Optional[str] = Field(None, description="The name of the persona of the preset.")
    human: str = Field(default=get_human_text(DEFAULT_HUMAN), description="The human of the preset.")
    human_name: Optional[str] = Field(None, description="The name of the human of the preset.")
    functions_schema: List[Dict] = Field(..., description="The functions schema of the preset.")


class ToolModel(SQLModel, table=True):
    # TODO move into database
    name: str = Field(..., description="The name of the function.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the function.", primary_key=True)
    tags: List[str] = Field(sa_column=Column(JSON), description="Metadata tags.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    source_code: Optional[str] = Field(..., description="The source code of the function.")

    json_schema: Dict = Field(default_factory=dict, sa_column=Column(JSON), description="The JSON schema of the function.")

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True


class AgentToolMap(SQLModel, table=True):
    # mapping between agents and tools
    agent_id: uuid.UUID = Field(..., description="The unique identifier of the agent.")
    tool_id: uuid.UUID = Field(..., description="The unique identifier of the tool.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the agent-tool map.", primary_key=True)


class PresetToolMap(SQLModel, table=True):
    # mapping between presets and tools
    preset_id: uuid.UUID = Field(..., description="The unique identifier of the preset.")
    tool_id: uuid.UUID = Field(..., description="The unique identifier of the tool.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the preset-tool map.", primary_key=True)


class AgentStateModel(BaseModel):
    id: uuid.UUID = Field(..., description="The unique identifier of the agent.")
    name: str = Field(..., description="The name of the agent.")
    description: Optional[str] = Field(None, description="The description of the agent.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the agent.")

    # timestamps
    # created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the agent was created.")
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


class CoreMemory(BaseModel):
    human: str = Field(..., description="Human element of the core memory.")
    persona: str = Field(..., description="Persona element of the core memory.")


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


class SourceModel(SQLModel, table=True):
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the source.")
    created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the source was created.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the source.", primary_key=True)
    description: Optional[str] = Field(None, description="The description of the source.")
    # embedding info
    # embedding_config: EmbeddingConfigModel = Field(..., description="The embedding configuration used by the source.")
    embedding_config: Optional[EmbeddingConfigModel] = Field(
        None, sa_column=Column(JSON), description="The embedding configuration used by the passage."
    )
    # NOTE: .metadata is a reserved attribute on SQLModel
    metadata_: Optional[dict] = Field(None, sa_column=Column(JSON), description="Metadata associated with the source.")


class JobStatus(str, Enum):
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobModel(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the job.", primary_key=True)
    # status: str = Field(default="created", description="The status of the job.")
    status: JobStatus = Field(default=JobStatus.created, description="The status of the job.", sa_column=Column(ChoiceType(JobStatus)))
    created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the job was created.")
    completed_at: Optional[datetime] = Field(None, description="The unix timestamp of when the job was completed.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the job.")
    metadata_: Optional[dict] = Field({}, sa_column=Column(JSON), description="The metadata of the job.")


class PassageModel(BaseModel):
    user_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the user associated with the passage.")
    agent_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the agent associated with the passage.")
    text: str = Field(..., description="The text of the passage.")
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfigModel] = Field(
        None, sa_column=Column(JSON), description="The embedding configuration used by the passage."
    )
    data_source: Optional[str] = Field(None, description="The data source of the passage.")
    doc_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the document associated with the passage.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the passage.", primary_key=True)
    metadata: Optional[Dict] = Field({}, description="The metadata of the passage.")


class DocumentModel(BaseModel):
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the document.")
    text: str = Field(..., description="The text of the document.")
    data_source: str = Field(..., description="The data source of the document.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the document.", primary_key=True)
    metadata: Optional[Dict] = Field({}, description="The metadata of the document.")
