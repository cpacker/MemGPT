from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.base import Base


class SourcesAgents(Base):
    """Agents can have zero to many sources"""

    __tablename__ = "sources_agents"

    agent_id: Mapped[String] = mapped_column(String, ForeignKey("agents.id"), primary_key=True)
    source_id: Mapped[String] = mapped_column(String, ForeignKey("sources.id"), primary_key=True)
