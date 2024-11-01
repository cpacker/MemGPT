from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.tag import AgentsTags as PydanticAgentsTags


class AgentsTags(SqlalchemyBase, OrganizationMixin):
    """Associates tags with agents, allowing agents to have multiple tags and supporting tag-based filtering."""

    __tablename__ = "tags_agents"
    __pydantic_model__ = PydanticAgentsTags
    __table_args__ = (UniqueConstraint("agent_id", "tag", name="unique_agent_tag"),)

    # The agent associated with this tag
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"), primary_key=True)

    # The name of the tag
    tag: Mapped[str] = mapped_column(String, nullable=False, doc="The name of the tag associated with the agent.")
