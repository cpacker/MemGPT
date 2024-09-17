from typing import TYPE_CHECKING, List, Optional, Type

from sqlalchemy import JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.block import Block
from memgpt.orm.message import Message
from memgpt.orm.mixins import OrganizationMixin
from memgpt.orm.organization import Organization
from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.schemas.agent import AgentState
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.llm_config import LLMConfig
from memgpt.orm.users_agents import UsersAgents
from memgpt.orm.sources_agents import SourcesAgents
from memgpt.orm.tools_agents import ToolsAgents
from memgpt.orm.blocks_agents import BlocksAgents

if TYPE_CHECKING:
    from pydantic import BaseModel

    from memgpt.orm.organization import Organization
    from memgpt.orm.passage import Passage
    from memgpt.orm.source import Source
    from memgpt.orm.tool import Tool
    from memgpt.orm.user import User


class Agent(SqlalchemyBase, OrganizationMixin):
    __tablename__ = "agent"
    __pydantic_model__ = AgentState
    __table_args__ = (
        UniqueConstraint(
            "_organization_id",
            "name",
            name="unique_agent_name_per_organization",
        ),
    )

    name: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="a human-readable identifier for an agent, non-unique.")
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The description of the agent.")
    system: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The system prompt used by the agent.")
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, doc="metadata for the agent.")
    llm_config: Mapped[Optional[LLMConfig]] = mapped_column(JSON, nullable=True, doc="the LLM backend configuration object for this agent.")
    embedding_config: Mapped[Optional[EmbeddingConfig]] = mapped_column(JSON, doc="the embedding configuration object for this agent.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="agents")
    users: Mapped[List["User"]] = relationship("User", back_populates="agents", secondary="users_agents", lazy="selectin")
    sources: Mapped[List["Source"]] = relationship("Source", secondary="sources_agents")
    tools: Mapped[List["Tool"]] = relationship("Tool", secondary="tools_agents", lazy="selectin")
    core_memory: Mapped[List["Block"]] = relationship("Block", secondary="blocks_agents", lazy="selectin")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="agent", lazy="selectin")
    passages: Mapped[List["Passage"]] = relationship("Passage", back_populates="agent", lazy="selectin")

    def to_pydantic(self) -> Type["BaseModel"]:
        """converts to the basic pydantic model counterpart"""
        state = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "description": self.description,
            "system": self.system,
            "metadata_": self.metadata_,
            "llm_config": self.llm_config,
            "embedding_config": self.embedding_config,
            "user_id": self.created_by_id,
            "tools": self.tools,
            "memory": {"memory": {b.name: b for b in self.core_memory}},
            "message_ids": self.messages,
        }
        return self.__pydantic_model__(**state)
