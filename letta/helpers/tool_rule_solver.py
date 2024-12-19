import json
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from letta.schemas.enums import ToolRuleType
from letta.schemas.tool_rule import (
    BaseToolRule,
    ChildToolRule,
    ConditionalToolRule,
    InitToolRule,
    TerminalToolRule,
)


class ToolRuleValidationError(Exception):
    """Custom exception for tool rule validation errors in ToolRulesSolver."""

    def __init__(self, message: str):
        super().__init__(f"ToolRuleValidationError: {message}")


class ToolRulesSolver(BaseModel):
    init_tool_rules: List[InitToolRule] = Field(
        default_factory=list, description="Initial tool rules to be used at the start of tool execution."
    )
    tool_rules: List[Union[ChildToolRule, ConditionalToolRule]] = Field(
        default_factory=list, description="Standard tool rules for controlling execution sequence and allowed transitions."
    )
    terminal_tool_rules: List[TerminalToolRule] = Field(
        default_factory=list, description="Terminal tool rules that end the agent loop if called."
    )
    last_tool_name: Optional[str] = Field(None, description="The most recent tool used, updated with each tool call.")

    def __init__(self, tool_rules: List[BaseToolRule], **kwargs):
        super().__init__(**kwargs)
        # Separate the provided tool rules into init, standard, and terminal categories
        for rule in tool_rules:
            if rule.type == ToolRuleType.run_first:
                assert isinstance(rule, InitToolRule)
                self.init_tool_rules.append(rule)
            elif rule.type == ToolRuleType.constrain_child_tools:
                assert isinstance(rule, ChildToolRule)
                self.tool_rules.append(rule)
            elif rule.type == ToolRuleType.conditional:
                assert isinstance(rule, ConditionalToolRule)
                self.validate_conditional_tool(rule)
                self.tool_rules.append(rule)
            elif rule.type == ToolRuleType.exit_loop:
                assert isinstance(rule, TerminalToolRule)
                self.terminal_tool_rules.append(rule)


    def update_tool_usage(self, tool_name: str):
        """Update the internal state to track the last tool called."""
        self.last_tool_name = tool_name

    def get_allowed_tool_names(self, error_on_empty: bool = False, last_function_response: Optional[str] = None) -> List[str]:
        """Get a list of tool names allowed based on the last tool called."""
        if self.last_tool_name is None:
            # Use initial tool rules if no tool has been called yet
            return [rule.tool_name for rule in self.init_tool_rules]
        else:
            # Find a matching ToolRule for the last tool used
            current_rule = next((rule for rule in self.tool_rules if rule.tool_name == self.last_tool_name), None)

            if current_rule is None:
                if error_on_empty:
                    raise ValueError(f"No tool rule found for {self.last_tool_name}")
                return []

            # If the current rule is a conditional tool rule, use the LLM response to
            # determine which child tool to use
            if isinstance(current_rule, ConditionalToolRule):
                if not last_function_response:
                    raise ValueError("Conditional tool rule requires an LLM response to determine which child tool to use")
                next_tool = self.evaluate_conditional_tool(current_rule, last_function_response)
                return [next_tool] if next_tool else []

            return current_rule.children if current_rule.children else []

    def is_terminal_tool(self, tool_name: str) -> bool:
        """Check if the tool is defined as a terminal tool in the terminal tool rules."""
        return any(rule.tool_name == tool_name for rule in self.terminal_tool_rules)

    def has_children_tools(self, tool_name):
        """Check if the tool has children tools"""
        return any(rule.tool_name == tool_name for rule in self.tool_rules)

    def validate_conditional_tool(self, rule: ConditionalToolRule):
        '''
        Validate a conditional tool rule

        Args:
            rule (ConditionalToolRule): The conditional tool rule to validate

        Raises:
            ToolRuleValidationError: If the rule is invalid
        '''
        if len(rule.child_output_mapping) == 0:
            raise ToolRuleValidationError("Conditional tool rule must have at least one child tool.")
        return True

    def evaluate_conditional_tool(self, tool: ConditionalToolRule, last_function_response: str) -> str:
        '''
        Parse function response to determine which child tool to use based on the mapping

        Args:
            tool (ConditionalToolRule): The conditional tool rule
            last_function_response (str): The function response in JSON format

        Returns:
            str: The name of the child tool to use next
        '''
        json_response = json.loads(last_function_response)
        function_output = json_response["message"]

        # Try to match the function output with a mapping key
        for key in tool.child_output_mapping:

            # Convert function output to match key type for comparison
            if isinstance(key, bool):
                typed_output = function_output.lower() == "true"
            elif isinstance(key, int):
                try:
                    typed_output = int(function_output)
                except (ValueError, TypeError):
                    continue
            elif isinstance(key, float):
                try:
                    typed_output = float(function_output)
                except (ValueError, TypeError):
                    continue
            else:  # string
                if function_output == "True" or function_output == "False":
                    typed_output = function_output.lower()
                elif function_output == "None":
                    typed_output = None
                else:
                    typed_output = function_output

            if typed_output == key:
                return tool.child_output_mapping[key]

        # If no match found, use default
        return tool.default_child
