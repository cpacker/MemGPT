from datetime import datetime
from typing import Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class UserBase(LettaBase):
    __id_prefix__ = "user"


class User(UserBase):
    """
    Representation of a user.

    Parameters:
        id (str): The unique identifier of the user.
        name (str): The name of the user.
        created_at (datetime): The creation date of the user.
    """

    id: str = UserBase.generate_id_field()
    organization_id: Optional[str] = Field(
        ..., description="The organization id of the user"
    )  # TODO: dont make optional, and pass in default org ID
    name: str = Field(..., description="The name of the user.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="The creation date of the user.")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="The update date of the user.")
    is_deleted: bool = Field(False, description="Whether this user is deleted or not.")


class UserCreate(UserBase):
    name: str = Field(..., description="The name of the user.")
    org_id: str = Field(..., description="The organization id of the user.")


class UserUpdate(UserBase):
    id: str = Field(..., description="The id of the user.")
    name: Optional[str] = Field(None, description="The name of the user.")
    org_id: Optional[str] = Field(None, description="The organization id of the user.")
