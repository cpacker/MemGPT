from typing import TYPE_CHECKING

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.mixins import AgentMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.schemas.message import Message as PydanticMessage

if TYPE_CHECKING:
    from memgpt.orm.agent import Agent


class Message(AgentMixin, SqlalchemyBase):
    """Text from an Agent or User that may include function call data."""

    __tablename__ = "message"
    __pydantic_model__ = PydanticMessage

    role: Mapped[str] = mapped_column(nullable=False, doc="The role of the user who created this message instance.")
    text: Mapped[str] = mapped_column(nullable=True, doc="The text of the message.")
    model: Mapped[str] = mapped_column(nullable=True, doc="Optional model name of the LLM that created this message.")
    name: Mapped[str] = mapped_column(nullable=True, doc="Optional name label of the LLM that created this message.")
    tool_calls: Mapped[dict] = mapped_column(JSON, nullable=True, doc="tool call results for this message.")
    tool_call_id: Mapped[str] = mapped_column(nullable=True, doc="Optional tool call ID for this message.")
    user_id: Mapped[str] = mapped_column(nullable=True, doc="The user ID of the user who created this message.")

    # relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="messages")
