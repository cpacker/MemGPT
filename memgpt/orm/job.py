from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.enums import JobStatus
from memgpt.orm.mixins import UserMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase

if TYPE_CHECKING:
    from memgpt.orm.user import User


class Job(UserMixin, SqlalchemyBase):
    """Jobs run in the background and are owned by a user.
    Typical jobs involve loading and processing sources etc.
    """

    __tablename__ = "job"

    status: Mapped[JobStatus] = mapped_column(default=JobStatus.created, doc="The current status of the job.")
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True, doc="The unix timestamp of when the job was completed.")
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, default=lambda: {}, doc="The metadata of the job.")

    # relationships
    user: Mapped["User"] = relationship("User", back_populates="jobs")
