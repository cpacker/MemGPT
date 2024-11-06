from datetime import datetime
from typing import Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase
from letta.utils import create_random_username, get_utc_time


class OrganizationBase(LettaBase):
    __id_prefix__ = "org"


class Organization(OrganizationBase):
    id: str = OrganizationBase.generate_id_field()
    name: str = Field(create_random_username(), description="The name of the organization.")
    created_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The creation date of the organization.")


class OrganizationCreate(OrganizationBase):
    name: Optional[str] = Field(None, description="The name of the organization.")
