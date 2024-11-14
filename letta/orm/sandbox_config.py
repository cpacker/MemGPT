from typing import TYPE_CHECKING, Dict, Optional

from sqlalchemy import JSON
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.sandbox_config import SandboxConfig as PydanticSandboxConfig
from letta.schemas.sandbox_config import (
    SandboxEnvironmentVariable as PydanticSandboxEnvironmentVariable,
)
from letta.schemas.sandbox_config import SandboxType

if TYPE_CHECKING:
    from letta.orm.organization import Organization


class SandboxConfig(SqlalchemyBase, OrganizationMixin):
    """ORM model for sandbox configurations with JSON storage for arbitrary config data."""

    __tablename__ = "sandbox_configs"
    __pydantic_model__ = PydanticSandboxConfig

    __table_args__ = (UniqueConstraint("type", "organization_id", name="uix_type_organization"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    type: Mapped[SandboxType] = mapped_column(SqlEnum(SandboxType), nullable=False, doc="The type of sandbox.")
    metadata_: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True, doc="Metadata associated with the sandbox.")
    config: Mapped[Dict] = mapped_column(JSON, nullable=False, doc="The JSON configuration data.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="sandbox_configs")


class SandboxEnvironmentVariable(SqlalchemyBase, OrganizationMixin):
    """ORM model for environment variables associated with sandboxes."""

    __tablename__ = "sandbox_environment_variables"
    __pydantic_model__ = PydanticSandboxEnvironmentVariable

    __table_args__ = (UniqueConstraint("key", "organization_id", name="uix_key_organization"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False, doc="The name of the environment variable.")
    value: Mapped[str] = mapped_column(String, nullable=False, doc="The value of the environment variable.")
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="An optional description of the environment variable.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="sandbox_environment_variables")
