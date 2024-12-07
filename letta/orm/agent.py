import uuid
from typing import TYPE_CHECKING, List, Optional, Type, Union

from sqlalchemy import JSON, String, TypeDecorator
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.block import Block
from letta.orm.message import Message
from letta.orm.mixins import OrganizationMixin
from letta.orm.organization import Organization
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.agent import AgentState, AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import ToolRuleType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import Memory
from letta.schemas.tool_rule import (
    ChildToolRule,
    InitToolRule,
    TerminalToolRule,
    ToolRule,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from letta.orm.organization import Organization
    from letta.orm.source import Source
    from letta.orm.tool import Tool


class LLMConfigColumn(TypeDecorator):
    """Custom type for storing LLMConfig as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            # return vars(value)
            if isinstance(value, LLMConfig):
                return value.model_dump()
        return value

    def process_result_value(self, value, dialect):
        if value:
            return LLMConfig(**value)
        return value


class EmbeddingConfigColumn(TypeDecorator):
    """Custom type for storing EmbeddingConfig as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            # return vars(value)
            if isinstance(value, EmbeddingConfig):
                return value.model_dump()
        return value

    def process_result_value(self, value, dialect):
        if value:
            return EmbeddingConfig(**value)
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
    def deserialize_tool_rule(data: dict) -> Union[ChildToolRule, InitToolRule, TerminalToolRule]:
        """Deserialize a dictionary to the appropriate ToolRule subclass based on the 'type'."""
        rule_type = ToolRuleType(data.get("type"))  # Remove 'type' field if it exists since it is a class var
        if rule_type == ToolRuleType.run_first:
            return InitToolRule(**data)
        elif rule_type == ToolRuleType.exit_loop:
            return TerminalToolRule(**data)
        elif rule_type == ToolRuleType.constrain_child_tools:
            rule = ChildToolRule(**data)
            return rule
        else:
            raise ValueError(f"Unknown tool rule type: {rule_type}")


class Agent(SqlalchemyBase, OrganizationMixin):
    __tablename__ = "agents"
    __pydantic_model__ = AgentState

    # agent generates its own id
    # TODO: We want to migrate all the ORM models to do this, so we will need to move this to the SqlalchemyBase
    # TODO: Move this in this PR? at the very end?
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"agent-{uuid.uuid4()}")

    # Descriptor fields
    agent_type: Mapped[Optional[AgentType]] = mapped_column(String, nullable=True, doc="The type of Agent")
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="a human-readable identifier for an agent, non-unique.")
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The description of the agent.")

    # System prompt
    system: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The system prompt used by the agent.")

    # In context memory
    # TODO: This should be a separate mapping table
    # This is dangerously flexible with the JSON type
    message_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True, doc="List of message IDs in in-context memory.")

    # Metadata and configs
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, doc="metadata for the agent.")
    llm_config: Mapped[Optional[LLMConfig]] = mapped_column(
        LLMConfigColumn, nullable=True, doc="the LLM backend configuration object for this agent."
    )
    embedding_config: Mapped[Optional[EmbeddingConfig]] = mapped_column(
        EmbeddingConfigColumn, doc="the embedding configuration object for this agent."
    )

    # Tool rules
    tool_rules: Mapped[Optional[List[ToolRule]]] = mapped_column(ToolRulesColumn, doc="the tool rules for this agent.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="agents")
    tools: Mapped[List["Tool"]] = relationship("Tool", secondary="tools_agents", lazy="joined")
    sources: Mapped[List["Source"]] = relationship("Source", secondary="sources_agents", lazy="joined")
    memory: Mapped[List["Block"]] = relationship("Block", secondary="blocks_agents", lazy="joined")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="agent",
        lazy="selectin",
        cascade="all, delete-orphan",  # Ensure messages are deleted when the agent is deleted
        passive_deletes=True,
    )

    # TODO: Add this back
    tags: Mapped[List["TagModel"]] = relationship(
        "TagModel", secondary="tags_agents", lazy="selectin", doc="Tags associated with the agent."
    )
    # passages: Mapped[List["Passage"]] = relationship("Passage", back_populates="agent", lazy="selectin")

    def to_pydantic(self) -> Type["BaseModel"]:
        """converts to the basic pydantic model counterpart"""
        state = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "message_ids": self.message_ids,
            "tools": self.tools,
            "sources": self.sources,
            "tags": [],  # TODO: Add this back in
            "tool_rules": self.tool_rules,
            "system": self.system,
            "agent_type": self.agent_type,
            "llm_config": self.llm_config,
            "embedding_config": self.embedding_config,
            "metadata_": self.metadata_,
            "memory": Memory(blocks=self.memory),
        }
        return self.__pydantic_model__(**state)
