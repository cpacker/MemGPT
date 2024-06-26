from sqlalchemy import ForeignKey, UUID as SQLUUID
from uuid import UUID
from sqlalchemy.orm import Mapped, mapped_column

from memgpt.orm.base import Base


class ToolsPresets(Base):
    """Tools can be used by zero to many Presets"""
    __tablename__ = 'tools_presets'

    _preset_id:Mapped[UUID] = mapped_column(SQLUUID, ForeignKey('preset._id'), primary_key=True)
    _tool_id:Mapped[UUID] = mapped_column(SQLUUID, ForeignKey('tool._id'), primary_key=True)