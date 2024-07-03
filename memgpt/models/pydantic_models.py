# tool imports
from uuid import UUID
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from memgpt.settings import settings
from memgpt.orm.enums import JobStatus
from memgpt.utils import get_human_text, get_persona_text, get_utc_time


class MemGPTUsageStatistics(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    step_count: int

class PersistedBase(BaseModel):
    """shared elements that all models coming from the ORM will support"""
    id: str = Field(description="The unique identifier of the object prefixed with the object type (Stripe pattern).")
    uuid: UUID = Field(description="The unique identifier of the object stored as a raw uuid (for legacy support).")
    deleted: Optional[bool] = Field(default=False, description="Is this record deleted? Used for universal soft deletes.")
    created_at: datetime = Field(description="The unix timestamp of when the object was created.")
    updated_at: datetime = Field(description="The unix timestamp of when the object was last updated.")
    created_by_id: Optional[str] = Field(description="The unique identifier of the user who created the object.")
    last_updated_by_id: Optional[str] = Field(description="The unique identifier of the user who last updated the object.")

class LLMConfigModel(BaseModel):
    # TODO: 🤮 don't default to a vendor! bug city!
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

class OrganizationSummary(PersistedBase):
    """An Organization interface with minimal references, good when only the link is needed"""
    name: str = Field(..., description="The name of the organization.")

class UserSummary(PersistedBase):
    """A User interface with minimal references, good when only the link is needed"""
    name: Optional[str] = Field(default=None, description="The name of the user.")
    email: Optional[str] = Field(default=None, description="The email of the user.")
    organization: Optional[OrganizationSummary] = Field(None, description="The organization this user belongs to.")

class PresetModel(PersistedBase):
    name: str = Field(description="The name of the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    system: str = Field(..., description="The system prompt of the preset.")
    # TODO: these should never default if the ORM manages defaults
    persona: str = Field(default=get_persona_text(settings.persona), description="The persona of the preset.")
    persona_name: Optional[str] = Field(None, description="The name of the persona of the preset.")
    human: str = Field(default=get_human_text(settings.human), description="The human of the preset.")
    human_name: Optional[str] = Field(None, description="The name of the human of the preset.")
    functions_schema: List[dict] = Field(..., description="The functions schema of the preset.")
    organization: Optional[OrganizationSummary] = Field(None, description="The organization this Preset belongs to.")

class ToolModel(PersistedBase):
    name: str = Field(..., description="The name of the function.")
    tags: List[str] = Field(description="Metadata tags.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    source_code: Optional[str] = Field(None, description="The source code of the function.")
    module: Optional[str] = Field(None, description="The module of the function.")

    json_schema: Dict = Field(default_factory=dict, description="The JSON schema of the function.")

    organization: Optional[OrganizationSummary] = Field(None, description="The organization this function belongs to.")

class AgentStateModel(PersistedBase):
    name: str = Field(..., description="The name of the agent.")
    description: Optional[str] = Field(None, description="The description of the agent.")

    # preset information
    tools: List[str] = Field(..., description="The tools used by the agent.")
    system: str = Field(..., description="The system prompt used by the agent.")

    # llm information
    llm_config: LLMConfigModel = Field(..., description="The LLM configuration used by the agent.")
    embedding_config: EmbeddingConfigModel = Field(..., description="The embedding configuration used by the agent.")

    # agent state
    state: Optional[Dict] = Field(None, description="The state of the agent.")
    metadata: Optional[Dict] = Field(None, description="The metadata of the agent.", alias="metadata_")


class CoreMemory(BaseModel):
    human: str = Field(..., description="Human element of the core memory.")
    persona: str = Field(..., description="Persona element of the core memory.")

class MemorySection(PersistedBase):
    """the common base for the legacy memory sections.
    This is going away in favor of MemoryModule dynamic sections.
    memgpt/memory.py
    """
    text: Optional[str] = Field(default=get_human_text(settings.human), description="The content to be added to this section of core memory.")
    name: str = Field(..., description="The name of the memory section.")
    organization: Optional[OrganizationSummary] = Field(None, description="The organization this memory belongs to.")

class HumanMemory(MemorySection):
    """Specifically for human, legacy"""


class PersonaModel(MemorySection):
    """Specifically for persona, legacy"""


class SourceModel(PersistedBase):
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")
    embedding_config: Optional[EmbeddingConfigModel] = Field(
        None, description="The embedding configuration used by the passage."
    )
    # NOTE: .metadata is a reserved attribute on SQLModel
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")

class JobModel(PersistedBase):
    status: JobStatus = Field(default=JobStatus.created, description="The status of the job.")
    completed_at: Optional[datetime] = Field(None, description="The unix timestamp of when the job was completed.")
    user: UserSummary = Field(description="The user associated with the job.")
    metadata_: Optional[dict] = Field({}, description="The metadata of the job.")

class PassageModel(PersistedBase):
    text: str = Field(..., description="The text of the passage.")
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfigModel] = Field(
        None, description="The embedding configuration used by the passage."
    )
    document: "DocumentModel" = Field(description="The document associated with the passage.")
    metadata_: Optional[dict] = Field({}, description="The metadata of the passage.")

class DocumentModel(PersistedBase):
    organization: OrganizationSummary = Field(description="The organization this document belongs to.")
    text: str = Field(..., description="The full text of the document.")
    data_source: str = Field(..., description="The data source of the document.")
    metadata_: Optional[Dict] = Field({}, description="The metadata of the document.")
