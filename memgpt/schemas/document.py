import uuid
from typing import Dict, Optional

from pydantic import BaseModel, Field


class DocumentModel(BaseModel):
    """Representation of a single document (broken up into `Passage` objects)"""

    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the document.")
    text: str = Field(..., description="The text of the document.")
    data_source: str = Field(..., description="The data source of the document.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the document.", primary_key=True)
    metadata: Optional[Dict] = Field({}, description="The metadata of the document.")
