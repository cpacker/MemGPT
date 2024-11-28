from typing import List, Union

from pydantic import Field

from letta.schemas.enums import ToolRuleType
from letta.schemas.letta_base import LettaBase


class BaseToolRule(LettaBase):
    __id_prefix__ = "tool_rule"
    tool_name: str = Field(..., description="The name of the tool. Must exist in the database for the user's organization.")
    type: ToolRuleType


class ChildToolRule(BaseToolRule):
    """
    A ToolRule represents a tool that can be invoked by the agent.
    """

    # type: str = Field("ToolRule")
    type: ToolRuleType = ToolRuleType.constrain_child_tools
    children: List[str] = Field(..., description="The children tools that can be invoked.")


class InitToolRule(BaseToolRule):
    """
    Represents the initial tool rule configuration.
    """

    # type: str = Field("InitToolRule")
    type: ToolRuleType = ToolRuleType.run_first


class TerminalToolRule(BaseToolRule):
    """
    Represents a terminal tool rule configuration where if this tool gets called, it must end the agent loop.
    """

    # type: str = Field("TerminalToolRule")
    type: ToolRuleType = ToolRuleType.exit_loop


ToolRule = Union[ChildToolRule, InitToolRule, TerminalToolRule]
