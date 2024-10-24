from typing import Dict, List, Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase
from letta.schemas.openai.chat_completions import ToolCall
from letta.services.organization_manager import OrganizationManager
from letta.services.user_manager import UserManager


class BaseTool(LettaBase):
    __id_prefix__ = "tool"

    # optional fields
    description: Optional[str] = Field(None, description="The description of the tool.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    module: Optional[str] = Field(None, description="The module of the function.")

    user_id: str = Field(..., description="The unique identifier of the user associated with the tool.")
    organization_id: str = Field(..., description="The unique identifier of the organization associated with the tool.")


class Tool(BaseTool):
    """
    Representation of a tool, which is a function that can be called by the agent.

    Parameters:
        id (str): The unique identifier of the tool.
        name (str): The name of the function.
        tags (List[str]): Metadata tags.
        source_code (str): The source code of the function.
        json_schema (Dict): The JSON schema of the function.

    """

    id: str = Field(..., description="The id of the tool.")

    name: str = Field(..., description="The name of the function.")
    tags: List[str] = Field(..., description="Metadata tags.")

    # code
    source_code: str = Field(..., description="The source code of the function.")
    json_schema: Dict = Field(default_factory=dict, description="The JSON schema of the function.")

    def to_dict(self):
        """
        Convert tool into OpenAI representation.
        """
        return vars(
            ToolCall(
                tool_id=self.id,
                tool_call_type="function",
                function=self.module,
            )
        )


class ToolCreate(LettaBase):
    user_id: str = Field(UserManager.DEFAULT_USER_ID, description="The user that this tool belongs to. Defaults to the default user ID.")
    organization_id: str = Field(
        OrganizationManager.DEFAULT_ORG_ID,
        description="The organization that this tool belongs to. Defaults to the default organization ID.",
    )
    name: Optional[str] = Field(None, description="The name of the function (auto-generated from source_code if not provided).")
    description: Optional[str] = Field(None, description="The description of the tool.")
    tags: List[str] = Field([], description="Metadata tags.")
    module: Optional[str] = Field(None, description="The source code of the function.")
    source_code: str = Field(..., description="The source code of the function.")
    source_type: str = Field(..., description="The source type of the function.")
    json_schema: Optional[Dict] = Field(
        None, description="The JSON schema of the function (auto-generated from source_code if not provided)"
    )
    terminal: Optional[bool] = Field(None, description="Whether the tool is a terminal tool (allow requesting heartbeats).")


class ToolUpdate(LettaBase):
    description: Optional[str] = Field(None, description="The description of the tool.")
    name: Optional[str] = Field(None, description="The name of the function.")
    tags: Optional[List[str]] = Field(None, description="Metadata tags.")
    module: Optional[str] = Field(None, description="The source code of the function.")
    source_code: Optional[str] = Field(None, description="The source code of the function.")
    json_schema: Optional[Dict] = Field(None, description="The JSON schema of the function.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
