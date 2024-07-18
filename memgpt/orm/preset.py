from typing import Optional, List, TYPE_CHECKING
from uuid import uuid4
from sqlalchemy import UniqueConstraint, JSON # TODO: jsonb for pg
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import OrganizationMixin, HumanMemoryTemplateMixin, PersonaMemoryTemplateMixin, SystemMemoryTemplateMixin

if TYPE_CHECKING:
    from memgpt.orm.organization import Organization
    from memgpt.orm.memory_templates import HumanMemoryTemplate, PersonaMemoryTemplate, SystemMemoryTemplate
    from memgpt.orm.source import Source
    from memgpt.orm.tool import Tool

class Preset(SqlalchemyBase, OrganizationMixin, HumanMemoryTemplateMixin, PersonaMemoryTemplateMixin, SystemMemoryTemplateMixin):
    """A preset represents a fixed starting point for an Agent, like a template of sorts.
    It is similar to OpenAI's concept of an `assistant`<https://platform.openai.com/docs/api-reference/assistants>_
    """
    __tablename__ = 'preset'
    __table_args__ = (
        UniqueConstraint(
            "_organization_id",
            "name",
            name="unique_name_organization",
        ),
    )

    name: Mapped[str] = mapped_column(doc="the name of the preset, must be unique within the org", nullable=False)
    description: Mapped[str] = mapped_column(nullable=True, doc="a human-readable description of the preset")
    system_name:Mapped[str] = mapped_column(doc="the name of the system message for the agent - DEPRECATED")
    human_name:Mapped[str] = mapped_column(doc="the name of the human message for the agent - DEPRECATED")
    persona_name:Mapped[str] = mapped_column(doc="the name of the persona message for the agent - DEPRECATED")
    # legacy JSON for support in sqlite and postgres. TODO: jsonb for pg
    functions_schema:Mapped[dict] = mapped_column(JSON, doc="the schema for the functions in the preset - DEPRECATED")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization",
                                                        back_populates="presets")
    sources: Mapped[List["Source"]] = relationship("Source", secondary="sources_presets")
    tools: Mapped[List["Tool"]] = relationship("Tool", secondary="tools_presets")
    human: Mapped["HumanMemoryTemplate"] = relationship("Human")
    persona: Mapped["PersonaMemoryTemplate"] = relationship("Persona")
    system: Mapped["SystemMemoryTemplate"] = relationship("System")


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # defaults for the names of the system, human, and persona messages
        for attr in ("system", "human", "persona"):
            attr_name = f"{attr}_name"
            if not getattr(self, attr_name):
                setattr(self, attr_name, f"{attr}_{self.name}_{str(uuid4())}")
