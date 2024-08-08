from typing import TYPE_CHECKING, Optional, Type, Union, List
from sqlalchemy import String, Integer, UUID as SQLUUID, ForeignKey
from sqlalchemy.orm import mapped_column, Mapped, relationship
from sqlalchemy.types import TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import _relation_getter, _relation_setter
from memgpt.schemas.block import Block as PydanticBlock, Human, Persona

if TYPE_CHECKING:
    from uuid import UUID
    from memgpt.orm.organization import Organization

class BlockValue(TypeDecorator):
    """block content can be a string or a list of strings, and we want to preserve that in the database
    This type will render a single string or a list of strings when deserialized,
    and will always store as a list of strings in the database.
    """

    impl = JSONB

    cache_ok = True

    def process_bind_param(self, value, dialect):
        """strings become a list with the string as the only element"""
        if isinstance(value, str):
            return [value]
        return value

    def process_result_value(self, value, dialect):
        """if the value is a list with a single element, return just that element"""
        if value and len(value) == 1:
            return value[0]
        return value


class Block(SqlalchemyBase):
    """Blocks are sections of the LLM context, representing a specific part of the total Memory"""
    __tablename__ = 'block'
    __pydantic_model__ = PydanticBlock

    name:Mapped[Optional[str]] = mapped_column(nullable=True, doc="the unique name that identifies a block in a human-readable way")
    description:Mapped[Optional[str]] = mapped_column(nullable=True, doc="a description of the block for context")
    label:Mapped[str] = mapped_column(doc="the type of memory block in use, ie 'human', 'persona', 'system'", primary_key=True)
    is_template:Mapped[bool] = mapped_column(doc="whether the block is a template (e.g. saved human/persona options as baselines for other templates)")
    value: Mapped[Optional[Union[List, str]]] = mapped_column(BlockValue, nullable=True, doc="Text content of the block for the respective section of core memory.")
    limit: Mapped[int] = mapped_column(Integer, default=2000, doc="Character limit of the block.")
    metadata_: Mapped[Optional[dict]] = mapped_column(JSONB, default={}, doc="arbitrary information related to the block.")

    # custom fkeys
    _organization_id: Mapped[Optional["UUID"]] = mapped_column(SQLUUID, ForeignKey("organization._id"),nullable=True, doc="the organization this block belongs to, if any")

    @property
    def organization_id(self) -> str:
        return _relation_getter(self, "organization")

    @organization_id.setter
    def organization_id(self, value: str) -> None:
        _relation_setter(self, "organization", value)

    # relationships
    organization:Mapped[Optional["Organization"]] = relationship("Organization")

    def to_pydantic(self) -> Type:
        match self.label:
            case "human":
                Schema = Human
            case "persona":
                Schema = Persona
            case _:
                Schema = PydanticBlock
        return Schema.model_validate(self)