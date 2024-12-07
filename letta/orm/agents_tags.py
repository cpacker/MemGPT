from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.base import Base


class AgentsTags(Base):
    __tablename__ = "agents_tags"
    __table_args__ = (UniqueConstraint("agent_id", "tag", name="unique_agent_tag"),)

    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"), primary_key=True)
    tag: Mapped[str] = mapped_column(String, nullable=False, doc="The name of the tag associated with the agent.")
