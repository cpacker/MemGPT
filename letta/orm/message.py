from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import Column, String, DateTime, Index, JSON, UniqueConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator, BINARY
import numpy as np
import base64

from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.orm.mixins import UserMixin, AgentMixin
from letta.schemas.message import Message as PydanticMessage

class Message(SqlalchemyBase, UserMixin, AgentMixin):
    """Defines data model for storing Message objects"""
    __tablename__ = "messages"
    __table_args__ = {"extend_existing": True}
    __pydantic_model__ = PydanticMessage
    
    id: Mapped[str] = mapped_column(primary_key=True, doc="Unique message identifier")
    role: Mapped[str] = mapped_column(doc="Message role (user/assistant/system/tool)")
    text: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Message content")
    model: Mapped[Optional[str]] = mapped_column(nullable=True, doc="LLM model used")
    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Name for multi-agent scenarios")
    tool_calls: Mapped[dict] = mapped_column(JSON, doc="Tool call information")
    tool_call_id: Mapped[Optional[str]] = mapped_column(nullable=True, doc="ID of the tool call")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="messages", lazy="selectin")
    # TODO: Add in after Agent ORM is created
    # agent: Mapped["Agent"] = relationship("Agent", back_populates="messages", lazy="selectin")

    __table_args__ = (
        Index("message_idx_user", "user_id", "agent_id"),
    )
