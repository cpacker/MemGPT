from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, String, func, text
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

    def _set_created_and_updated_by_fields(self, actor_id: str) -> None:
        """Populate created_by_id and last_updated_by_id based on actor."""
        if not self.created_by_id:
            self.created_by_id = actor_id
        # Always set the last_updated_by_id when updating
        self.last_updated_by_id = actor_id

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
        return mapped_column(String, nullable=True)

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
        return prop_value

    def _user_id_setter(self, prop: str, value: str) -> None:
        """returns the user id for the specified property"""
        full_prop = f"_{prop}_by_id"
        if not value:
            setattr(self, full_prop, None)
            return
        # Safety check
        prefix, id_ = value.split("-", 1)
        assert prefix == "user", f"{prefix} is not a valid id prefix for a user id"

        # Set the full value
        setattr(self, full_prop, value)
