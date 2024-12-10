import uuid

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import AgentMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase


class AgentsTags(SqlalchemyBase, AgentMixin):
    __tablename__ = "agents_tags"
    __table_args__ = (UniqueConstraint("agent_id", "tag", name="unique_agent_tag"),)

    # agent generates its own id
    # TODO: We want to migrate all the ORM models to do this, so we will need to move this to the SqlalchemyBase
    # TODO: Move this in this PR? at the very end?
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"agents_tags-{uuid.uuid4()}")

    tag: Mapped[str] = mapped_column(String, nullable=False, doc="The name of the tag associated with the agent.")

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="tags")
