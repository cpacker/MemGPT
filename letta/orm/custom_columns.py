import base64
from typing import List, Union

import numpy as np
from sqlalchemy import JSON
from sqlalchemy.types import BINARY, TypeDecorator

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import ToolRuleType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.openai.chat_completions import ToolCall, ToolCallFunction
from letta.schemas.tool_rule import ChildToolRule, ConditionalToolRule, InitToolRule, TerminalToolRule


class EmbeddingConfigColumn(TypeDecorator):
    """Custom type for storing EmbeddingConfig as JSON."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value and isinstance(value, EmbeddingConfig):
            return value.model_dump()
        return value

    def process_result_value(self, value, dialect):
        if value:
            return EmbeddingConfig(**value)
        return value


class LLMConfigColumn(TypeDecorator):
    """Custom type for storing LLMConfig as JSON."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value and isinstance(value, LLMConfig):
            return value.model_dump()
        return value

    def process_result_value(self, value, dialect):
        if value:
            return LLMConfig(**value)
        return value


class ToolRulesColumn(TypeDecorator):
    """Custom type for storing a list of ToolRules as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        """Convert a list of ToolRules to JSON-serializable format."""
        if value:
            data = [rule.model_dump() for rule in value]
            for d in data:
                d["type"] = d["type"].value

            for d in data:
                assert not (d["type"] == "ToolRule" and "children" not in d), "ToolRule does not have children field"
            return data
        return value

    def process_result_value(self, value, dialect) -> List[Union[ChildToolRule, InitToolRule, TerminalToolRule]]:
        """Convert JSON back to a list of ToolRules."""
        if value:
            return [self.deserialize_tool_rule(rule_data) for rule_data in value]
        return value

    @staticmethod
    def deserialize_tool_rule(data: dict) -> Union[ChildToolRule, InitToolRule, TerminalToolRule, ConditionalToolRule]:
        """Deserialize a dictionary to the appropriate ToolRule subclass based on the 'type'."""
        rule_type = ToolRuleType(data.get("type"))  # Remove 'type' field if it exists since it is a class var
        if rule_type == ToolRuleType.run_first:
            return InitToolRule(**data)
        elif rule_type == ToolRuleType.exit_loop:
            return TerminalToolRule(**data)
        elif rule_type == ToolRuleType.constrain_child_tools:
            rule = ChildToolRule(**data)
            return rule
        elif rule_type == ToolRuleType.conditional:
            rule = ConditionalToolRule(**data)
            return rule
        else:
            raise ValueError(f"Unknown tool rule type: {rule_type}")


class ToolCallColumn(TypeDecorator):

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            values = []
            for v in value:
                if isinstance(v, ToolCall):
                    values.append(v.model_dump())
                else:
                    values.append(v)
            return values

        return value

    def process_result_value(self, value, dialect):
        if value:
            tools = []
            for tool_value in value:
                if "function" in tool_value:
                    tool_call_function = ToolCallFunction(**tool_value["function"])
                    del tool_value["function"]
                else:
                    tool_call_function = None
                tools.append(ToolCall(function=tool_call_function, **tool_value))
            return tools
        return value


class CommonVector(TypeDecorator):
    """Common type for representing vectors in SQLite"""

    impl = BINARY
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(BINARY())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, list):
            value = np.array(value, dtype=np.float32)
        return base64.b64encode(value.tobytes())

    def process_result_value(self, value, dialect):
        if not value:
            return value
        if dialect.name == "sqlite":
            value = base64.b64decode(value)
        return np.frombuffer(value, dtype=np.float32)
