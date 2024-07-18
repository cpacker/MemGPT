import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from memgpt.schemas.embedding_config import EmbeddingConfig


class PassageModel(BaseModel):
    user_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the user associated with the passage.")
    agent_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the agent associated with the passage.")
    text: str = Field(..., description="The text of the passage.")
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
    data_source: Optional[str] = Field(None, description="The data source of the passage.")
    doc_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the document associated with the passage.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the passage.", primary_key=True)
    metadata: Optional[Dict] = Field({}, description="The metadata of the passage.")
