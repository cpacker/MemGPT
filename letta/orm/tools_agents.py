from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm import Base


class ToolsAgents(Base):
    """Agents can have one or many tools associated with them."""

    __tablename__ = "tools_agents"
    __table_args__ = (UniqueConstraint("agent_id", "tool_id", name="unique_agent_tool"),)

    # Each agent must have unique tool names
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True)
    tool_id: Mapped[str] = mapped_column(String, ForeignKey("tools.id", ondelete="CASCADE"), primary_key=True)
