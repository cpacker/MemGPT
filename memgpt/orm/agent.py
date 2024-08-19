from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.sources_agents import SourcesAgents
from memgpt.orm.tools_agents import ToolsAgents
from memgpt.orm.mixins import OrganizationMixin
from memgpt.schemas.llm_config import LLMConfig
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.orm.blocks_agents import BlocksAgents
from memgpt.orm.block import Block

if TYPE_CHECKING:
    from memgpt.orm.organization import Organization
    from memgpt.orm.source import Source
    from memgpt.orm.user import User
    from memgpt.orm.tool import Tool

class Agent(SqlalchemyBase, OrganizationMixin):
    __tablename__ = 'agent'

    name:Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="a human-readable identifier for an agent, non-unique.")

    persona: Mapped[str] = mapped_column(doc="the persona text for the agent, current state.")

    # TODO: reconcile this with persona,human etc AND make this structured via pydantic!
    # TODO: these are vague and need to be more specific and explained. WTF is state vs _metadata?
    state: Mapped[dict] = mapped_column(JSON, doc="the state of the agent.")
    _metadata: Mapped[dict] = mapped_column(JSON, doc="metadata for the agent.")
    # todo: this doesn't allign with 1:M agents to users!
    human: Mapped[str] = mapped_column(doc="the human text for the agent and the current user, current state.")
    preset: Mapped[str] = mapped_column(doc="the preset text for the agent, current state.")

    llm_config: Mapped[LLMConfig] = mapped_column(JSON, doc="the LLM backend configuration object for this agent.")
    embedding_config: Mapped[EmbeddingConfig] = mapped_column(JSON, doc="the embedding configuration object for this agent.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="agents")
    users: Mapped[List["User"]] = relationship("User",
                                               back_populates="agents",
                                               secondary="users_agents")
    sources: Mapped[List["Source"]] = relationship("Source", secondary="sources_agents")
    tools: Mapped[List["Tool"]] = relationship("Tool", secondary="tools_agents")

    core_memory: Mapped[List["Block"]] = relationship("Block", secondary="blocks_agents")