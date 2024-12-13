from typing import Optional

from sqlalchemy import JSON, TypeDecorator
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import AgentMixin, OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.openai.chat_completions import ToolCall, ToolCallFunction


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


class Message(SqlalchemyBase, OrganizationMixin, AgentMixin):
    """Defines data model for storing Message objects"""

    __tablename__ = "messages"
    __table_args__ = {"extend_existing": True}
    __pydantic_model__ = PydanticMessage

    id: Mapped[str] = mapped_column(primary_key=True, doc="Unique message identifier")
    role: Mapped[str] = mapped_column(doc="Message role (user/assistant/system/tool)")
    text: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Message content")
    model: Mapped[Optional[str]] = mapped_column(nullable=True, doc="LLM model used")
    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Name for multi-agent scenarios")
    tool_calls: Mapped[ToolCall] = mapped_column(ToolCallColumn, doc="Tool call information")
    tool_call_id: Mapped[Optional[str]] = mapped_column(nullable=True, doc="ID of the tool call")

    # Relationships
    # TODO: Add in after Agent ORM is created
    # agent: Mapped["Agent"] = relationship("Agent", back_populates="messages", lazy="selectin")
    organization: Mapped["Organization"] = relationship("Organization", back_populates="messages", lazy="selectin")
