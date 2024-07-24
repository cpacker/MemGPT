from uuid import UUID

from sqlalchemy import UUID as SQLUUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from memgpt.orm.base import Base


class SourcesPresets(Base):
    """Sources can be used by zero to many Presets"""

    __tablename__ = "sources_presets"

    _preset_id: Mapped[UUID] = mapped_column(SQLUUID, ForeignKey("preset._id"), primary_key=True)
    _source_id: Mapped[UUID] = mapped_column(SQLUUID, ForeignKey("source._id"), primary_key=True)
