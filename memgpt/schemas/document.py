from typing import Dict, Optional

from pydantic import Field

from memgpt.schemas.memgpt_base import MemGPTBase


class DocumentBase(MemGPTBase):
    """Base class for document schemas"""

    __id_prefix__ = "doc"


class Document(DocumentBase):
    """Representation of a single document (broken up into `Passage` objects)"""

    id: str = DocumentBase.generate_id_field()
    text: str = Field(..., description="The text of the document.")
    source_id: str = Field(..., description="The unique identifier of the source associated with the document.")
    user_id: str = Field(description="The unique identifier of the user associated with the document.")
    metadata_: Optional[Dict] = Field({}, description="The metadata of the document.")
