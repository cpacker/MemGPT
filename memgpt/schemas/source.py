from typing import Optional

from pydantic import Field

from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.memgpt_base import MemGPTBase


class BaseSource(MemGPTBase):
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


class Source(BaseSource):
    id: str = BaseSource.generate_id_field()
    name: str = Field(..., description="The name of the source.")


class SourceUpdate(BaseSource):
    id: str = BaseSource.generate_id_field()  # TODO not sure if this is correct?
    name: Optional[str] = Field(None, description="The name of the source.")


class SourceAttach(BaseSource):
    agent_id: str = Field(..., description="The ID of the agent to attach the source to.")
    source_id: Optional[str] = Field(None, description="The ID of the source.")
    # source_name: Optional[str] = Field(None, description="The name of the source.")


class SourceDetach(BaseSource):
    agent_id: str = Field(..., description="The ID of the agent to detach the source from.")
    source_id: Optional[str] = Field(None, description="The ID of the source.")
    # source_name: Optional[str] = Field(None, description="The name of the source.")
