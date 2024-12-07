from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.base import Base


class SourcesAgents(Base):
    """Agents can have zero to many sources"""

    __tablename__ = "sources_agents"
    __table_args__ = (UniqueConstraint("agent_id", "source_id", name="unique_agent_source"),)  # Note the comma

    agent_id: Mapped[String] = mapped_column(String, ForeignKey("agents.id"), primary_key=True)
    source_id: Mapped[String] = mapped_column(String, ForeignKey("sources.id"), primary_key=True)
