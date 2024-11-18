from datetime import datetime
from typing import Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class FileMetadataBase(LettaBase):
    """Base class for FileMetadata schemas"""

    __id_prefix__ = "file"


class FileMetadata(FileMetadataBase):
    """Representation of a single FileMetadata"""

    id: str = FileMetadataBase.generate_id_field()
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the document.")
    source_id: str = Field(..., description="The unique identifier of the source associated with the document.")
    file_name: Optional[str] = Field(None, description="The name of the file.")
    file_path: Optional[str] = Field(None, description="The path to the file.")
    file_type: Optional[str] = Field(None, description="The type of the file (MIME type).")
    file_size: Optional[int] = Field(None, description="The size of the file in bytes.")
    file_creation_date: Optional[str] = Field(None, description="The creation date of the file.")
    file_last_modified_date: Optional[str] = Field(None, description="The last modified date of the file.")

    # orm metadata, optional fields
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="The creation date of the file.")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="The update date of the file.")
    is_deleted: bool = Field(False, description="Whether this file is deleted or not.")
