import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from memgpt.schemas.embedding_config import EmbeddingConfig


class CommonPassagePartial(BaseModel):
    # associated user/agent
    user_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the user associated with the passage.")
    agent_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the agent associated with the passage.")

    # origin data source
    data_source: Optional[str] = Field(None, description="The data source of the passage.")

    # document association
    doc_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the document associated with the passage.")
    metadata: Optional[Dict] = Field({}, description="The metadata of the passage.")


class Passage(CommonPassagePartial):
    passage_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the passage.", primary_key=True)

    # passage text
    text: str = Field(..., description="The text of the passage.")

    # embeddings
    embedding: List[float] = Field(..., description="The embedding of the passage.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the passage.")


class PassageCreate(CommonPassagePartial):
    text: str = Field(..., description="The text of the passage.")

    # optionally provide embeddings
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")


class PassageUpdate(PassageCreate):
    passage_id: uuid.UUID = Field(..., description="The unique identifier of the passage.")
    text: Optional[str] = Field(None, description="The text of the passage.")

    # optionally provide embeddings
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
