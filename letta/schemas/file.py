from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from letta.schemas.letta_base import LettaBase
from letta.utils import get_utc_time


class FileBase(LettaBase):
    """Base class for document schemas"""

    __id_prefix__ = "doc"


class File(FileBase):
    """Representation of a single document (broken up into `Passage` objects)"""

    id: str = FileBase.generate_id_field()
    user_id: str = Field(description="The unique identifier of the user associated with the document.")
    source_id: str = Field(..., description="The unique identifier of the source associated with the document.")
    metadata_: Optional[Dict] = Field({}, description="The metadata of the document.")
    created_at: datetime = Field(default_factory=get_utc_time, description="The creation date of the passage.")

    class Config:
        extra = "allow"


class PaginatedListFilesResponse(BaseModel):
    files: List[File]
    next_cursor: Optional[str] = None  # The cursor for fetching the next page, if any
