from datetime import datetime
from typing import Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.letta_base import LettaBase
from letta.utils import get_utc_time


class BaseSource(LettaBase):
    """
    Shared attributes accourss all source schemas.
    """

    __id_prefix__ = "source"
    description: Optional[str] = Field(None, description="The description of the source.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
    # NOTE: .metadata is a reserved attribute on SQLModel
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")


class SourceCreate(BaseSource):
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")


class Source(BaseSource):
    """
    Representation of a source, which is a collection of files and passages.

    Parameters:
        id (str): The ID of the source
        name (str): The name of the source.
        embedding_config (EmbeddingConfig): The embedding configuration used by the source.
        created_at (datetime): The creation date of the source.
        user_id (str): The ID of the user that created the source.
        metadata_ (dict): Metadata associated with the source.
        description (str): The description of the source.
    """

    id: str = BaseSource.generate_id_field()
    name: str = Field(..., description="The name of the source.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the source.")
    created_at: datetime = Field(default_factory=get_utc_time, description="The creation date of the source.")
    user_id: str = Field(..., description="The ID of the user that created the source.")


class SourceUpdate(BaseSource):
    id: str = Field(..., description="The ID of the source.")
    name: Optional[str] = Field(None, description="The name of the source.")


class UploadFileToSourceRequest(BaseModel):
    file: UploadFile = Field(..., description="The file to upload.")


class UploadFileToSourceResponse(BaseModel):
    source: Source = Field(..., description="The source the file was uploaded to.")
    added_passages: int = Field(..., description="The number of passages added to the source.")
    added_documents: int = Field(..., description="The number of files added to the source.")
