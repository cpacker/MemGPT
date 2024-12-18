from typing import Dict, List, Optional, Union
from collections import deque

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

        # Validate the tool rules to ensure they form a DAG
        if not self.validate_tool_rules():
            raise ToolRuleValidationError("Tool rules does not have a path from Init to Terminal.")

    def update_tool_usage(self, tool_name: str):
        """Update the internal state to track the last tool called."""
        self.last_tool_name = tool_name

    def get_allowed_tool_names(self, error_on_empty: bool = False) -> List[str]:
        """Get a list of tool names allowed based on the last tool called."""
        if self.last_tool_name is None:
            # Use initial tool rules if no tool has been called yet
            return [rule.tool_name for rule in self.init_tool_rules]
        else:
            # Find a matching ToolRule for the last tool used
            current_rule = next((rule for rule in self.tool_rules if rule.tool_name == self.last_tool_name), None)

            # Return children which must exist on ToolRule
            if current_rule:
                return current_rule.children

            # Default to empty if no rule matches
            message = "User provided tool rules and execution state resolved to no more possible tool calls."
            if error_on_empty:
                raise RuntimeError(message)
            else:
                # warnings.warn(message)
                return []

    def is_terminal_tool(self, tool_name: str) -> bool:
        """Check if the tool is defined as a terminal tool in the terminal tool rules."""
        return any(rule.tool_name == tool_name for rule in self.terminal_tool_rules)

    def has_children_tools(self, tool_name):
        """Check if the tool has children tools"""
        return any(rule.tool_name == tool_name for rule in self.tool_rules)

    def validate_conditional_tool(self, rule: ConditionalToolRule):
        if rule.children is None or len(rule.children) == 0:
            raise ToolRuleValidationError("Conditional tool rule must have at least one child tool.")
        if len(rule.children) != len(rule.child_output_mapping):
            raise ToolRuleValidationError("Conditional tool rule must have a child output mapping for each child tool.")
        if set(rule.children) != set(rule.child_output_mapping.values()):
            raise ToolRuleValidationError("Conditional tool rule must have a child output mapping for each child tool.")
        return True

    def validate_tool_rules(self) -> bool:
        """
        Validate that there exists a path from every init tool to a terminal tool.
        Returns True if valid (path exists), otherwise False.
        """
        # Build adjacency list for the tool graph
        adjacency_list: Dict[str, List[str]] = {rule.tool_name: rule.children for rule in self.tool_rules}

        init_tool_names = {rule.tool_name for rule in self.init_tool_rules}
        terminal_tool_names = {rule.tool_name for rule in self.terminal_tool_rules}

        # Initial checks
        if len(init_tool_names) == 0:
            if len(terminal_tool_names) + len(self.tool_rules) > 0:
                return False  # No init tools defined
            else:
                return True   # No tool rules
        if len(terminal_tool_names) == 0:
            if len(adjacency_list) > 0:
                return False  # No terminal tools defined
            else:
                return True   # Only init tools

        # Define BFS helper function to find path to terminal tool
        def has_path_to_terminal(start_tool: str) -> bool:
            visited = set()
            queue = deque([start_tool])
            visited.add(start_tool)

            while queue:
                current_tool = queue.popleft()
                if current_tool in terminal_tool_names:
                    return True

                for child in adjacency_list.get(current_tool, []):
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
            return False

        # Check if each init tool has a path to a terminal tool
        for init_tool_name in init_tool_names:
            if not has_path_to_terminal(init_tool_name):
                return False

        return True  # All init tools have paths to terminal tools
