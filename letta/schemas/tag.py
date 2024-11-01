from datetime import datetime

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class AgentsTags(LettaBase):
    """
    Schema representing the relationship between tags and agents.

    Parameters:
        agent_id (str): The ID of the associated agent.
        tag_id (str): The ID of the associated tag.
        tag_name (str): The name of the tag.
        created_at (datetime): The date this relationship was created.
    """

    agent_id: str = Field(..., description="The ID of the associated agent.")
    tag: str = Field(..., description="The name of the tag.")
    organization_id: str = Field(..., description="The organization this belongs to.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="The creation date of the association.")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="The update date of the tag.")
    is_deleted: bool = Field(False, description="Whether this tag is deleted or not.")
