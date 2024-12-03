from uuid import UUID

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.base import Base


def is_valid_uuid4(uuid_string: str) -> bool:
    """Check if a string is a valid UUID4."""
    try:
        uuid_obj = UUID(uuid_string)
        return uuid_obj.version == 4
    except ValueError:
        return False


class OrganizationMixin(Base):
    """Mixin for models that belong to an organization."""

    __abstract__ = True

    organization_id: Mapped[str] = mapped_column(String, ForeignKey("organizations.id"))


class UserMixin(Base):
    """Mixin for models that belong to a user."""

    __abstract__ = True

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))


class SourceMixin(Base):
    """Mixin for models (e.g. file) that belong to a source."""

    __abstract__ = True

    source_id: Mapped[str] = mapped_column(String, ForeignKey("sources.id"))


class SandboxConfigMixin(Base):
    """Mixin for models that belong to a SandboxConfig."""

    __abstract__ = True

    sandbox_config_id: Mapped[str] = mapped_column(String, ForeignKey("sandbox_configs.id"))
