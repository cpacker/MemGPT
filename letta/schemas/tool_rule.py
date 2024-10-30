from typing import List

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class BaseToolRule(LettaBase):
    __id_prefix__ = "tool_rule"
    tool_name: str = Field(..., description="The name of the tool. Must exist in the database for the user's organization.")


class ToolRule(BaseToolRule):
    type: str = Field("ToolRule")
    children: List[str] = Field(..., description="The children tools that can be invoked.")


class InitToolRule(BaseToolRule):
    type: str = Field("InitToolRule")
    """Represents the initial tool rule configuration."""


class TerminalToolRule(BaseToolRule):
    type: str = Field("TerminalToolRule")
    """Represents a terminal tool rule configuration where if this tool gets called, it must end the agent loop."""
