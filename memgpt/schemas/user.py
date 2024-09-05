from datetime import datetime
from typing import Optional

from pydantic import Field

from memgpt.schemas.memgpt_base import MemGPTBase


class UserBase(MemGPTBase):
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
    name: str = Field(..., description="The name of the user.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="The creation date of the user.")


class UserCreate(UserBase):
    name: Optional[str] = Field(None, description="The name of the user.")
