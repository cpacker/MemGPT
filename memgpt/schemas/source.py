import uuid
from typing import Optional

from pydantic import BaseModel, Field

from memgpt.schemas.embedding_config import EmbeddingConfig


class Source(BaseModel):
    source_id: uuid.UUID = Field(..., description="The id of the source.")
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
    # NOTE: .metadata is a reserved attribute on SQLModel
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")


class SourceUpdate(BaseModel):
    source_id: uuid.UUID = Field(..., description="The id of the source.")
    name: Optional[str] = Field(None, description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")


class SourceCreate(BaseModel):
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")
