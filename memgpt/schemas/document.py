import uuid
from typing import Dict, Optional

from pydantic import Field

from memgpt.schemas.memgpt_base import MemGPTBase


class DocumentModel(MemGPTBase):
    """Representation of a single document (broken up into `Passage` objects)"""

    __id_prefix__ = "doc"
    id: str = MemGPTBase.generate_id_field()
    text: str = Field(..., description="The text of the document.")
    data_source: str = Field(..., description="The data source of the document.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the document.", primary_key=True)
    metadata: Optional[Dict] = Field({}, description="The metadata of the document.")
