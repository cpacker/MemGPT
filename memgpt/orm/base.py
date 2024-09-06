from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import UUID as SQLUUID
from sqlalchemy import Boolean, DateTime, func, text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declarative_mixin,
    declared_attr,
    mapped_column,
)


class Base(DeclarativeBase):
    """absolute base for sqlalchemy classes"""


@declarative_mixin
class CommonSqlalchemyMetaMixins(Base):
    __abstract__ = True

    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now(), server_onupdate=func.now())
    is_deleted: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))

    @declared_attr
    def _created_by_id(cls):
        return cls._user_by_id()

    @declared_attr
    def _last_updated_by_id(cls):
        return cls._user_by_id()

    @classmethod
    def _user_by_id(cls):
        """a flexible non-constrained record of a user.
        This way users can get added, deleted etc without history freaking out
        """
        return mapped_column(SQLUUID(), nullable=True)

    @property
    def last_updated_by_id(self) -> Optional[str]:
        return self._user_id_getter("last_updated")

    @last_updated_by_id.setter
    def last_updated_by_id(self, value: str) -> None:
        self._user_id_setter("last_updated", value)

    @property
    def created_by_id(self) -> Optional[str]:
        return self._user_id_getter("created")

    @created_by_id.setter
    def created_by_id(self, value: str) -> None:
        self._user_id_setter("created", value)

    def _user_id_getter(self, prop: str) -> Optional[str]:
        """returns the user id for the specified property"""
        full_prop = f"_{prop}_by_id"
        prop_value = getattr(self, full_prop, None)
        if not prop_value:
            return
        return f"user-{prop_value}"

    def _user_id_setter(self, prop: str, value: str) -> None:
        """returns the user id for the specified property"""
        full_prop = f"_{prop}_by_id"
        if not value:
            setattr(self, full_prop, None)
            return
        prefix, id_ = value.split("-", 1)
        assert prefix == "user", f"{prefix} is not a valid id prefix for a user id"
        setattr(self, full_prop, UUID(id_))
