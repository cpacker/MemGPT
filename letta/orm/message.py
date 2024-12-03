from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import Column, String, DateTime, Index, JSON, UniqueConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator, BINARY
import numpy as np
import base64

from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.orm.mixins import UserMixin, AgentMixin, FileMixin

if TYPE_CHECKING:
    from letta.orm.user import User
    from letta.orm.agent import Agent
    from letta.orm.file import File

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

class Message(SqlalchemyBase, UserMixin, AgentMixin):
    """Defines data model for storing Message objects"""
    __tablename__ = "messages"
    __table_args__ = {"extend_existing": True}

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
    agent: Mapped["Agent"] = relationship("Agent", back_populates="messages", lazy="selectin")

    __table_args__ = (
        Index("message_idx_user", "user_id", "agent_id"),
    )

class Passage(SqlalchemyBase, UserMixin, AgentMixin, FileMixin):
    """Defines data model for storing Passages"""
    __tablename__ = "passages"
    __table_args__ = {"extend_existing": True}

    id: Mapped[str] = mapped_column(primary_key=True, doc="Unique passage identifier")
    text: Mapped[str] = mapped_column(doc="Passage text content")
    source_id: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Source identifier")
    embedding: Mapped[bytes] = mapped_column(CommonVector, doc="Vector embedding")
    embedding_config: Mapped[dict] = mapped_column(JSON, doc="Embedding configuration")
    metadata_: Mapped[dict] = mapped_column(JSON, doc="Additional metadata")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="passages", lazy="selectin")
    agent: Mapped[Optional["Agent"]] = relationship("Agent", back_populates="passages", lazy="selectin")
    file: Mapped[Optional["File"]] = relationship("File", back_populates="passages", lazy="selectin")

    __table_args__ = (
        Index("passage_idx_user", "user_id", "agent_id", "file_id"),
    )