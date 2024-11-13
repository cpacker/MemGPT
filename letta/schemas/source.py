from datetime import datetime
from typing import Optional

from pydantic import Field

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.letta_base import LettaBase


class BaseSource(LettaBase):
    """
    Shared attributes accourss all source schemas.
    """

    __id_prefix__ = "source"


class Source(BaseSource):
    """
    Representation of a source, which is a collection of files and passages.

    Parameters:
        id (str): The ID of the source
        name (str): The name of the source.
        embedding_config (EmbeddingConfig): The embedding configuration used by the source.
        user_id (str): The ID of the user that created the source.
        metadata_ (dict): Metadata associated with the source.
        description (str): The description of the source.
    """

    id: str = BaseSource.generate_id_field()
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the source.")
    organization_id: Optional[str] = Field(None, description="The ID of the organization that created the source.")
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")

    # metadata fields
    created_by_id: Optional[str] = Field(None, description="The id of the user that made this Tool.")
    last_updated_by_id: Optional[str] = Field(None, description="The id of the user that made this Tool.")
    created_at: Optional[datetime] = Field(None, description="The timestamp when the source was created.")
    updated_at: Optional[datetime] = Field(None, description="The timestamp when the source was last updated.")


class SourceCreate(BaseSource):
    """
    Schema for creating a new Source.
    """

    # required
    name: str = Field(..., description="The name of the source.")
    # TODO: @matt, make this required after shub makes the FE changes
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the source.")

    # optional
    description: Optional[str] = Field(None, description="The description of the source.")
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")


class SourceUpdate(BaseSource):
    """
    Schema for updating an existing Source.
    """

    name: Optional[str] = Field(None, description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the source.")
