import warnings

import pytest

from letta.helpers import ToolRulesSolver
from letta.schemas.tool_rule import InitToolRule, TerminalToolRule, ToolRule

# Constants for tool names used in the tests
START_TOOL = "start_tool"
PREP_TOOL = "prep_tool"
NEXT_TOOL = "next_tool"
HELPER_TOOL = "helper_tool"
FINAL_TOOL = "final_tool"
END_TOOL = "end_tool"
UNRECOGNIZED_TOOL = "unrecognized_tool"


def test_get_allowed_tool_names_with_init_rules():
    # Setup: Initial tool rule configuration
    init_rule_1 = InitToolRule(tool_name=START_TOOL)
    init_rule_2 = InitToolRule(tool_name=PREP_TOOL)
    solver = ToolRulesSolver(init_tool_rules=[init_rule_1, init_rule_2], tool_rules=[], terminal_tool_rules=[])

    # Action: Get allowed tool names when no tool has been called
    allowed_tools = solver.get_allowed_tool_names()

    # Assert: Both init tools should be allowed initially
    assert allowed_tools == [START_TOOL, PREP_TOOL], "Should allow only InitToolRule tools at the start"


def test_get_allowed_tool_names_with_subsequent_rule():
    # Setup: Tool rule sequence
    init_rule = InitToolRule(tool_name=START_TOOL)
    rule_1 = ToolRule(tool_name=START_TOOL, children=[NEXT_TOOL, HELPER_TOOL])
    solver = ToolRulesSolver(init_tool_rules=[init_rule], tool_rules=[rule_1], terminal_tool_rules=[])

    # Action: Update usage and get allowed tools
    solver.update_tool_usage(START_TOOL)
    allowed_tools = solver.get_allowed_tool_names()

    # Assert: Only children of "start_tool" should be allowed
    assert allowed_tools == [NEXT_TOOL, HELPER_TOOL], "Should allow only children of the last tool used"


def test_is_terminal_tool():
    # Setup: Terminal tool rule configuration
    init_rule = InitToolRule(tool_name=START_TOOL)
    terminal_rule = TerminalToolRule(tool_name=END_TOOL)
    solver = ToolRulesSolver(init_tool_rules=[init_rule], tool_rules=[], terminal_tool_rules=[terminal_rule])

    # Action & Assert: Verify terminal and non-terminal tools
    assert solver.is_terminal_tool(END_TOOL) is True, "Should recognize 'end_tool' as a terminal tool"
    assert solver.is_terminal_tool(START_TOOL) is False, "Should not recognize 'start_tool' as a terminal tool"


def test_get_allowed_tool_names_no_matching_rule_warning():
    # Setup: Tool rules with no matching rule for the last tool
    init_rule = InitToolRule(tool_name=START_TOOL)
    solver = ToolRulesSolver(init_tool_rules=[init_rule], tool_rules=[], terminal_tool_rules=[])

    # Action: Set last tool to an unrecognized tool and check warnings
    solver.update_tool_usage(UNRECOGNIZED_TOOL)

    with warnings.catch_warnings(record=True) as w:
        allowed_tools = solver.get_allowed_tool_names()

        # Assert: Expecting a warning and an empty list of allowed tools
        assert len(w) == 1, "Expected a warning for no matching rule"
        assert "resolved to no more possible tool calls" in str(w[-1].message)
        assert allowed_tools == [], "Should return an empty list if no matching rule"


def test_get_allowed_tool_names_no_matching_rule_error():
    # Setup: Tool rules with no matching rule for the last tool
    init_rule = InitToolRule(tool_name=START_TOOL)
    solver = ToolRulesSolver(init_tool_rules=[init_rule], tool_rules=[], terminal_tool_rules=[])

    # Action & Assert: Set last tool to an unrecognized tool and expect RuntimeError when error_on_empty=True
    solver.update_tool_usage(UNRECOGNIZED_TOOL)
    with pytest.raises(RuntimeError, match="resolved to no more possible tool calls"):
        solver.get_allowed_tool_names(error_on_empty=True)


def test_update_tool_usage_and_get_allowed_tool_names_combined():
    # Setup: More complex rule chaining
    init_rule = InitToolRule(tool_name=START_TOOL)
    rule_1 = ToolRule(tool_name=START_TOOL, children=[NEXT_TOOL])
    rule_2 = ToolRule(tool_name=NEXT_TOOL, children=[FINAL_TOOL])
    terminal_rule = TerminalToolRule(tool_name=FINAL_TOOL)
    solver = ToolRulesSolver(init_tool_rules=[init_rule], tool_rules=[rule_1, rule_2], terminal_tool_rules=[terminal_rule])

    # Step 1: Initially allowed tools
    assert solver.get_allowed_tool_names() == [START_TOOL], "Initial allowed tool should be 'start_tool'"

    # Step 2: After using 'start_tool'
    solver.update_tool_usage(START_TOOL)
    assert solver.get_allowed_tool_names() == [NEXT_TOOL], "After 'start_tool', should allow 'next_tool'"

    # Step 3: After using 'next_tool'
    solver.update_tool_usage(NEXT_TOOL)
    assert solver.get_allowed_tool_names() == [FINAL_TOOL], "After 'next_tool', should allow 'final_tool'"

    # Step 4: 'final_tool' should be terminal
    assert solver.is_terminal_tool(FINAL_TOOL) is True, "Should recognize 'final_tool' as terminal"
