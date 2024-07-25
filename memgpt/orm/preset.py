from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, UniqueConstraint  # TODO: jsonb for pg
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.mixins import OrganizationMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase

if TYPE_CHECKING:
    from memgpt.orm.organization import Organization
    from memgpt.orm.source import Source
    from memgpt.orm.tool import Tool


class Preset(SqlalchemyBase, OrganizationMixin):
    """A preset represents a fixed starting point for an Agent, like a template of sorts.
    It is similar to OpenAI's concept of an `assistant`<https://platform.openai.com/docs/api-reference/assistants>_
    """

    __tablename__ = "preset"
    __table_args__ = (
        UniqueConstraint(
            "_organization_id",
            "name",
            name="unique_name_organization",
        ),
    )

    name: Mapped[str] = mapped_column(doc="the name of the preset, must be unique within the org", nullable=False)
    description: Mapped[str] = mapped_column(nullable=True, doc="a human-readable description of the preset")

    ## TODO: these are unclear - human vs human_name for example, what and why?
    system: Mapped[Optional[str]] = mapped_column(doc="the current system message for the agent.")
    human: Mapped[str] = mapped_column(doc="the current human message for the agent.")
    human_name: Mapped[str] = mapped_column(doc="the name of the human message for the agent - DEPRECATED")
    persona: Mapped[str] = mapped_column(doc="the current persona message for the agent.")
    persona_name: Mapped[str] = mapped_column(doc="the name of the persona message for the agent - DEPRECATED")
    # legacy JSON for support in sqlite and postgres. TODO: jsonb for pg
    functions_schema: Mapped[dict] = mapped_column(JSON, doc="the schema for the functions in the preset - DEPRECATED")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="presets")
    sources: Mapped[List["Source"]] = relationship("Source", secondary="sources_presets")
    tools: Mapped[List["Tool"]] = relationship("Tool", secondary="tools_presets")
