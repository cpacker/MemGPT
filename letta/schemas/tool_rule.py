from typing import Dict, List, Union

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

    type: ToolRuleType = ToolRuleType.constrain_child_tools
    children: List[str] = Field(..., description="The children tools that can be invoked.")


class ConditionalToolRule(BaseToolRule):
    """
    A ToolRule that conditionally maps to different child tools based on the output.
    """
    type: ToolRuleType = ToolRuleType.conditional
    default_child: str = Field(..., description="The default child tool to be called")
    child_output_mapping: Dict[Union[bool, str, int], str] = Field(..., description="The output case to check for mapping")
    children: List[str] = Field(..., description="The child tool to call when output matches the case")
    throw_error: bool = Field(default=False, description="Whether to throw an error when output doesn't match any case")


class InitToolRule(BaseToolRule):
    """
    Represents the initial tool rule configuration.
    """

    type: ToolRuleType = ToolRuleType.run_first


class TerminalToolRule(BaseToolRule):
    """
    Represents a terminal tool rule configuration where if this tool gets called, it must end the agent loop.
    """

    type: ToolRuleType = ToolRuleType.exit_loop


ToolRule = Union[ChildToolRule, InitToolRule, TerminalToolRule, ConditionalToolRule]
