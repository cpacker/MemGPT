from typing import Optional, TYPE_CHECKING
from pydantic import EmailStr
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Mapped, relationship, mapped_column

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
if TYPE_CHECKING:
    from memgpt.orm.user import User
    from sqlalchemy.orm.session import Session

class Organization(SqlalchemyBase):
    """The highest level of the object tree. All Entities belong to one and only one Organization."""
    __tablename__ = "organization"
    name:Mapped[Optional[str]] = mapped_column(nullable=True, doc="The display name of the organization.")

    # relationships
    users: Mapped["User"] = relationship("User", back_populates="organization")
    agents: Mapped["Agent"] = relationship("Agent", back_populates="organization")

    @classmethod
    def default(cls, db_session:"Session") -> "Organization":
        """Get the default org, or create it if it doesn't exist."""
        try:
            return db_session.query(cls).one().scalar()
        except NoResultFound:
            return cls(name="Default Organization").create(db_session)

