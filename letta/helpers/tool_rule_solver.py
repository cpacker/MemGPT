from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from letta.schemas.tool_rule import (
    BaseToolRule,
    InitToolRule,
    TerminalToolRule,
    ToolRule,
)


class ToolRuleValidationError(Exception):
    """Custom exception for tool rule validation errors in ToolRulesSolver."""

    def __init__(self, message: str):
        super().__init__(f"ToolRuleValidationError: {message}")


class ToolRulesSolver(BaseModel):
    init_tool_rules: List[InitToolRule] = Field(
        default_factory=list, description="Initial tool rules to be used at the start of tool execution."
    )
    tool_rules: List[ToolRule] = Field(
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
            if isinstance(rule, InitToolRule):
                self.init_tool_rules.append(rule)
            elif isinstance(rule, ToolRule):
                self.tool_rules.append(rule)
            elif isinstance(rule, TerminalToolRule):
                self.terminal_tool_rules.append(rule)

        # Validate the tool rules to ensure they form a DAG
        if not self.validate_tool_rules():
            raise ToolRuleValidationError("Tool rules contain cycles, which are not allowed in a valid configuration.")

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

    def validate_tool_rules(self) -> bool:
        """
        Validate that the tool rules define a directed acyclic graph (DAG).
        Returns True if valid (no cycles), otherwise False.
        """
        # Build adjacency list for the tool graph
        adjacency_list: Dict[str, List[str]] = {rule.tool_name: rule.children for rule in self.tool_rules}

        # Track visited nodes
        visited: Set[str] = set()
        path_stack: Set[str] = set()

        # Define DFS helper function
        def dfs(tool_name: str) -> bool:
            if tool_name in path_stack:
                return False  # Cycle detected
            if tool_name in visited:
                return True  # Already validated

            # Mark the node as visited in the current path
            path_stack.add(tool_name)
            for child in adjacency_list.get(tool_name, []):
                if not dfs(child):
                    return False  # Cycle detected in DFS
            path_stack.remove(tool_name)  # Remove from current path
            visited.add(tool_name)
            return True

        # Run DFS from each tool in `tool_rules`
        for rule in self.tool_rules:
            if rule.tool_name not in visited:
                if not dfs(rule.tool_name):
                    return False  # Cycle found, invalid tool rules

        return True  # No cycles, valid DAG
