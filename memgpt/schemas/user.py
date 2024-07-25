from datetime import datetime

from pydantic import Field

from memgpt.schemas.memgpt_base import MemGPTBase


class UserBase(MemGPTBase):
    __id_prefix__ = "user"


class User(UserBase):
    id: str = MemGPTBase.generate_id_field()
    name: str = Field(..., description="The name of the user.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="The creation date of the user.")


class UserCreate(UserBase):
    name: str = Field(..., description="The name of the user.")
