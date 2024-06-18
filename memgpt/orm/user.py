from typing import Optional, TYPE_CHECKING
from pydantic import EmailStr
from sqlalchemy import String
from sqlalchemy.orm import Mapped, relationship, mapped_column

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import OrganizationMixin

class User(SqlalchemyBase, OrganizationMixin):
    """User ORM class"""
    __tablename__ = "user"

    name:Mapped[Optional[str]] = mapped_column(nullable=True, doc="The display name of the user.")
    email:Mapped[Optional[EmailStr]] = mapped_column(String,
                                                     nullable=True,
                                                     doc="The email address of the user. Uninforced at this time.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")

