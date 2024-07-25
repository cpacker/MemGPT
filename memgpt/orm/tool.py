from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.enums import ToolSourceType
from memgpt.orm.mixins import OrganizationMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase

if TYPE_CHECKING:
    pass


class Tool(SqlalchemyBase, OrganizationMixin):
    """Represents an available tool that the LLM can invoke.

    NOTE: polymorphic inheritance makes more sense here as a TODO. We want a superset of tools
    that are always available, and a subset scoped to the organization. Alternatively, we could use the apply_access_predicate to build
    more granular permissions.
    """

    __tablename__ = "tool"

    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The display name of the tool.")
    # TODO: this needs to be a lookup table to have any value
    tags: Mapped[List] = mapped_column(JSON, doc="Metadata tags used to filter tools.")
    source_type: Mapped[ToolSourceType] = mapped_column(String, doc="The type of the source code.", default=ToolSourceType.json)
    source_code: Mapped[Optional[str]] = mapped_column(
        String, doc="The source code of the function if provided.", default=None, nullable=True
    )
    json_schema: Mapped[dict] = mapped_column(JSON, default=lambda: {}, doc="The OAI compatable JSON schema of the function.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="tools")
