from datetime import datetime

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class AgentsTagsBase(LettaBase):
    __id_prefix__ = "agents_tags"


class AgentsTags(AgentsTagsBase):
    """
    Schema representing the relationship between tags and agents.

    Parameters:
        agent_id (str): The ID of the associated agent.
        tag_id (str): The ID of the associated tag.
        tag_name (str): The name of the tag.
        created_at (datetime): The date this relationship was created.
    """

    id: str = Field(..., description="The internal id.")
    agent_id: str = Field(..., description="The ID of the associated agent.")
    tag: str = Field(..., description="The name of the tag.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="The creation date of the association.")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="The update date of the tag.")
    is_deleted: bool = Field(False, description="Whether this tag is deleted or not.")


class AgentsTagsCreate(AgentsTagsBase):
    tag: str = Field(..., description="The tag name.")
