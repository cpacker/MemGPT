from typing import TYPE_CHECKING, Optional, Type

from sqlalchemy import JSON, BigInteger, Integer, UniqueConstraint, event
from sqlalchemy.orm import Mapped, attributes, mapped_column, relationship

from letta.constants import CORE_MEMORY_BLOCK_CHAR_LIMIT
from letta.orm.blocks_agents import BlocksAgents
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.block import Human, Persona

if TYPE_CHECKING:
    from letta.orm import Organization


class Block(OrganizationMixin, SqlalchemyBase):
    """Blocks are sections of the LLM context, representing a specific part of the total Memory"""

    __tablename__ = "block"
    __pydantic_model__ = PydanticBlock
    # This may seem redundant, but is necessary for the BlocksAgents composite FK relationship
    __table_args__ = (UniqueConstraint("id", "label", name="unique_block_id_label"),)

    template_name: Mapped[Optional[str]] = mapped_column(
        nullable=True, doc="the unique name that identifies a block in a human-readable way"
    )
    description: Mapped[Optional[str]] = mapped_column(nullable=True, doc="a description of the block for context")
    label: Mapped[str] = mapped_column(doc="the type of memory block in use, ie 'human', 'persona', 'system'")
    is_template: Mapped[bool] = mapped_column(
        doc="whether the block is a template (e.g. saved human/persona options as baselines for other templates)", default=False
    )
    value: Mapped[str] = mapped_column(doc="Text content of the block for the respective section of core memory.")
    limit: Mapped[BigInteger] = mapped_column(Integer, default=CORE_MEMORY_BLOCK_CHAR_LIMIT, doc="Character limit of the block.")
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


@event.listens_for(Block, "after_update")  # Changed from 'before_update'
def block_before_update(mapper, connection, target):
    """Handle updating BlocksAgents when a block's label changes."""
    label_history = attributes.get_history(target, "label")
    if not label_history.has_changes():
        return

    blocks_agents = BlocksAgents.__table__
    connection.execute(
        blocks_agents.update()
        .where(blocks_agents.c.block_id == target.id, blocks_agents.c.block_label == label_history.deleted[0])
        .values(block_label=label_history.added[0])
    )


@event.listens_for(Block, "before_insert")
@event.listens_for(Block, "before_update")
def validate_value_length(mapper, connection, target):
    """Ensure the value length does not exceed the limit."""
    if target.value and len(target.value) > target.limit:
        raise ValueError(
            f"Value length ({len(target.value)}) exceeds the limit ({target.limit}) for block with label '{target.label}' and id '{target.id}'."
        )
