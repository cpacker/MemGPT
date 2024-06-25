from typing import Optional
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import UserMixin


class Token(SqlalchemyBase, UserMixin):
    __tablename__ = 'token'

    hash:Mapped[str] = mapped_column(String, doc="the secured one-way hash of the token")
    name:Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="a name to identify the token")

    user: Mapped["User"] = relationship("User", back_populates="tokens")