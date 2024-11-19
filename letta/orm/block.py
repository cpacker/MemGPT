from typing import TYPE_CHECKING, Optional, Type

from sqlalchemy import JSON, BigInteger, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.block import Human, Persona

if TYPE_CHECKING:
    from letta.orm.organization import Organization


class Block(OrganizationMixin, SqlalchemyBase):
    """Blocks are sections of the LLM context, representing a specific part of the total Memory"""

    __tablename__ = "block"
    __pydantic_model__ = PydanticBlock

    template_name: Mapped[Optional[str]] = mapped_column(
        nullable=True, doc="the unique name that identifies a block in a human-readable way"
    )
    description: Mapped[Optional[str]] = mapped_column(nullable=True, doc="a description of the block for context")
    label: Mapped[str] = mapped_column(doc="the type of memory block in use, ie 'human', 'persona', 'system'")
    is_template: Mapped[bool] = mapped_column(
        doc="whether the block is a template (e.g. saved human/persona options as baselines for other templates)", default=False
    )
    value: Mapped[str] = mapped_column(doc="Text content of the block for the respective section of core memory.")
    limit: Mapped[BigInteger] = mapped_column(Integer, default=2000, doc="Character limit of the block.")
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, default={}, doc="arbitrary information related to the block.")

    # relationships
    organization: Mapped[Optional["Organization"]] = relationship("Organization")

    def to_pydantic(self) -> Type:
        match self.label:
            case "human":
                Schema = Human
            case "persona":
                Schema = Persona
            case _:
                Schema = PydanticBlock
        return Schema.model_validate(self)
