import uuid
from typing import Dict, List, Optional

from pydantic import Field

from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.memgpt_base import MemGPTBase


class PassageBase(MemGPTBase):
    __id_prefix__ = "passage"

    # associated user/agent
    user_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the user associated with the passage.")
    agent_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the agent associated with the passage.")

    # origin data source
    data_source: Optional[str] = Field(None, description="The data source of the passage.")

    # document association
    doc_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the document associated with the passage.")
    metadata: Optional[Dict] = Field({}, description="The metadata of the passage.")


class Passage(PassageBase):
    id: str = MemGPTBase.generate_id_field()

    # passage text
    text: str = Field(..., description="The text of the passage.")

    # embeddings
    embedding: List[float] = Field(..., description="The embedding of the passage.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the passage.")


class PassageCreate(PassageBase):
    text: str = Field(..., description="The text of the passage.")

    # optionally provide embeddings
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")


class PassageUpdate(PassageCreate):
    id: str = Field(..., description="The unique identifier of the passage.")
    text: Optional[str] = Field(None, description="The text of the passage.")

    # optionally provide embeddings
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
