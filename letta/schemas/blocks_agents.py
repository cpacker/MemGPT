from datetime import datetime
from typing import Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class BlocksAgentsBase(LettaBase):
    __id_prefix__ = "blocks_agents"


class BlocksAgents(BlocksAgentsBase):
    """
    Schema representing the relationship between blocks and agents.

    Parameters:
        agent_id (str): The ID of the associated agent.
        block_id (str): The ID of the associated block.
        block_label (str): The label of the block.
        created_at (datetime): The date this relationship was created.
        updated_at (datetime): The date this relationship was last updated.
        is_deleted (bool): Whether this block-agent relationship is deleted or not.
    """

    id: str = BlocksAgentsBase.generate_id_field()
    agent_id: str = Field(..., description="The ID of the associated agent.")
    block_id: str = Field(..., description="The ID of the associated block.")
    block_label: str = Field(..., description="The label of the block.")
    created_at: Optional[datetime] = Field(None, description="The creation date of the association.")
    updated_at: Optional[datetime] = Field(None, description="The update date of the association.")
    is_deleted: bool = Field(False, description="Whether this block-agent relationship is deleted or not.")
