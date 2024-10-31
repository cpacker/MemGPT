from typing import Optional
from uuid import UUID

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.base import Base
from letta.orm.errors import MalformedIdError


def is_valid_uuid4(uuid_string: str) -> bool:
    """Check if a string is a valid UUID4."""
    try:
        uuid_obj = UUID(uuid_string)
        return uuid_obj.version == 4
    except ValueError:
        return False


def _relation_getter(instance: "Base", prop: str) -> Optional[str]:
    """Get relation and return id with prefix as a string."""
    prefix = prop.replace("_", "")
    formatted_prop = f"_{prop}_id"
    try:
        id_ = getattr(instance, formatted_prop)  # Get the string id directly
        return f"{prefix}-{id_}"
    except AttributeError:
        return None


def _relation_setter(instance: "Base", prop: str, value: str) -> None:
    """Set relation using the id with prefix, ensuring the id is a valid UUIDv4."""
    formatted_prop = f"_{prop}_id"
    prefix = prop.replace("_", "")
    if not value:
        setattr(instance, formatted_prop, None)
        return
    try:
        found_prefix, id_ = value.split("-", 1)
    except ValueError as e:
        raise MalformedIdError(f"{value} is not a valid ID.") from e

    # Ensure prefix matches
    assert found_prefix == prefix, f"{found_prefix} is not a valid id prefix, expecting {prefix}"

    # Validate that the id is a valid UUID4 string
    if not is_valid_uuid4(id_):
        raise MalformedIdError(f"Hash segment of {value} is not a valid UUID4")

    setattr(instance, formatted_prop, id_)  # Store id as a string


class OrganizationMixin(Base):
    """Mixin for models that belong to an organization."""

    __abstract__ = True

    _organization_id: Mapped[str] = mapped_column(String, ForeignKey("organization._id"))

    @property
    def organization_id(self) -> str:
        return _relation_getter(self, "organization")

    @organization_id.setter
    def organization_id(self, value: str) -> None:
        _relation_setter(self, "organization", value)


class UserMixin(Base):
    """Mixin for models that belong to a user."""

    __abstract__ = True

    _user_id: Mapped[str] = mapped_column(String, ForeignKey("user._id"))

    @property
    def user_id(self) -> str:
        return _relation_getter(self, "user")

    @user_id.setter
    def user_id(self, value: str) -> None:
        _relation_setter(self, "user", value)
