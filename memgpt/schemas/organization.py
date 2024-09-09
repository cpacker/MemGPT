from pydantic import Field

from memgpt.schemas.memgpt_base import MemGPTBase


class BaseOrganization(MemGPTBase, validate_assignment=True):
    """Blocks are sections of the LLM context, representing a specific part of the total Memory"""

    __id_prefix__ = "organization"
    __sqlalchemy_model__ = "Organization"


class Organization(BaseOrganization):
    """An Organization interface with minimal references, good when only the link is needed"""

    id: str = Field(..., description="The unique id of the organization.")
    name: str = Field(..., description="The name of the organization.")
