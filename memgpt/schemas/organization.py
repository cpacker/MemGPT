from datetime import datetime
from typing import Optional

from pydantic import Field

from memgpt.schemas.memgpt_base import MemGPTBase


class OrganizationBase(MemGPTBase):
    __id_prefix__ = "org"


class Organization(OrganizationBase):
    id: str = OrganizationBase.generate_id_field()
    name: str = Field(..., description="The name of the organization.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="The creation date of the user.")


class OrganizationCreate(OrganizationBase):
    name: Optional[str] = Field(None, description="The name of the organization.")
