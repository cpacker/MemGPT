from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.orm.mixins import OrganizationMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase

if TYPE_CHECKING:
    from memgpt.orm.organization import Organization


class MemoryTemplate(SqlalchemyBase, OrganizationMixin):
    """Memory templates define the structure and starting point for a given memory type."""

    __tablename__ = "memory_template"

    name: Mapped[str] = mapped_column(doc="the unique name that identifies a memory template")
    description: Mapped[str] = mapped_column(doc="a description of the memory template")
    type: Mapped[str] = mapped_column(doc="the type of memory template in use")
    text: Mapped[str] = mapped_column(doc="the starting memory text provided for the template")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization")

    __mapper_args__ = {
        "polymorphic_identity": "generic",
        "polymorphic_on": "type",
    }


class HumanMemoryTemplate(MemoryTemplate):
    """Template for the structured 'human' section of core memory.
    Note: will be migrated to dynamic memory templates in the future.
    """

    __mapper_args__ = {
        "polymorphic_identity": "human",
    }


class PersonaMemoryTemplate(MemoryTemplate):
    """Template for the structured 'persona' section of core memory.
    Note: will be migrated to dynamic memory templates in the future.
    """

    __mapper_args__ = {
        "polymorphic_identity": "persona",
    }


class SystemMemoryTemplate(MemoryTemplate):
    """Template for the structured 'system' section of core memory.
    Note: will be migrated to dynamic memory templates in the future.
    """

    __mapper_args__ = {
        "polymorphic_identity": "system",
    }
