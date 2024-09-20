from datetime import datetime

from pydantic import Field

from memgpt.schemas.memgpt_base import MemGPTBase


class OrganizationBase(MemGPTBase):
    __id_prefix__ = "org"


class Organization(OrganizationBase):
    id: str = OrganizationBase.generate_id_field()
    created_at: datetime = Field(default_factory=datetime.utcnow, description="The creation date of the user.")
