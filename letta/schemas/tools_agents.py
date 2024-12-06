from datetime import datetime
from typing import Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class ToolsAgentsBase(LettaBase):
    __id_prefix__ = "tools_agents"


class ToolsAgents(ToolsAgentsBase):
    """
    Schema representing the relationship between tools and agents.

    Parameters:
        agent_id (str): The ID of the associated agent.
        tool_id (str): The ID of the associated tool.
        tool_name (str): The name of the tool.
        created_at (datetime): The date this relationship was created.
        updated_at (datetime): The date this relationship was last updated.
        is_deleted (bool): Whether this tool-agent relationship is deleted or not.
    """

    id: str = ToolsAgentsBase.generate_id_field()
    agent_id: str = Field(..., description="The ID of the associated agent.")
    tool_id: str = Field(..., description="The ID of the associated tool.")
    tool_name: str = Field(..., description="The name of the tool.")
    created_at: Optional[datetime] = Field(None, description="The creation date of the association.")
    updated_at: Optional[datetime] = Field(None, description="The update date of the association.")
    is_deleted: bool = Field(False, description="Whether this tool-agent relationship is deleted or not.")
