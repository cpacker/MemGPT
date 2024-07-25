from uuid import UUID

from sqlalchemy import UUID as SQLUUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from memgpt.orm.base import Base


class ToolsAgents(Base):
    """Agents can have zero to many tools"""

    __tablename__ = "tools_agents"

    _agent_id: Mapped[UUID] = mapped_column(SQLUUID, ForeignKey("agent._id"), primary_key=True)
    _tool_id: Mapped[UUID] = mapped_column(SQLUUID, ForeignKey("tool._id"), primary_key=True)
