from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.agents_tags import AgentsTags as PydanticAgentsTags

if TYPE_CHECKING:
    from letta.orm.organization import Organization


class AgentsTags(SqlalchemyBase, OrganizationMixin):
    """Associates tags with agents, allowing agents to have multiple tags and supporting tag-based filtering."""

    __tablename__ = "agents_tags"
    __pydantic_model__ = PydanticAgentsTags
    __table_args__ = (UniqueConstraint("agent_id", "tag", name="unique_agent_tag"),)

    # The agent associated with this tag
    agent_id = mapped_column(String, ForeignKey("agents.id"), primary_key=True)

    # The name of the tag
    tag: Mapped[str] = mapped_column(String, nullable=False, doc="The name of the tag associated with the agent.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="agents_tags")
