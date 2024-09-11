from typing import TYPE_CHECKING, List, Optional, Type, Union

from sqlalchemy import JSON, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from memgpt.settings import settings
from memgpt.orm.mixins import OrganizationMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.organization import Organization
from memgpt.schemas.block import Block as PydanticBlock
from memgpt.schemas.block import Human, Persona
import memgpt.utils as utils


if TYPE_CHECKING:
    from memgpt.orm.organization import Organization
    from sqlalchemy.orm import Session


class BlockValue(TypeDecorator):
    """block content can be a string or a list of strings, and we want to preserve that in the database
    This type will render a single string or a list of strings when deserialized,
    and will always store as a list of strings in the database.
    """

    impl = JSON

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


class Block(OrganizationMixin, SqlalchemyBase):
    """Blocks are sections of the LLM context, representing a specific part of the total Memory"""

    __tablename__ = "block"
    __pydantic_model__ = PydanticBlock
    __table_args__ = (
        UniqueConstraint(
            "_organization_id",
            "name",
            name="unique_block_name_per_organization",
        ),
    )

    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="the unique name that identifies a block in a human-readable way")
    description: Mapped[Optional[str]] = mapped_column(nullable=True, doc="a description of the block for context")
    label: Mapped[str] = mapped_column(doc="the type of memory block in use, ie 'human', 'persona', 'system'", primary_key=True)
    is_template: Mapped[bool] = mapped_column(
        doc="whether the block is a template (e.g. saved human/persona options as baselines for other templates)"
    )
    value: Mapped[Optional[Union[List, str]]] = mapped_column(
        BlockValue, nullable=True, doc="Text content of the block for the respective section of core memory."
    )
    limit: Mapped[int] = mapped_column(Integer, default=2000, doc="Character limit of the block.")
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

    @classmethod
    def load_default_blocks(cls, db_session: "Session") -> None:
        """populates the db with default blocks"""
        org = Organization.default(db_session)
        sql_blocks = []
        for scope in ("human", "persona"):
            list_files = getattr(utils, f"list_{scope}_files")
            get_text = getattr(utils, f"get_{scope}_text")
            for file in list_files():
                sql_blocks.append(
                    dict(
                        _organization_id=org._id,
                        name=file.stem,
                        label=scope,
                        value=get_text(file.stem),
                        is_template=True,
                    )
                )
        match settings.backend.name:
            case "sqlite_chroma":
                from sqlalchemy.dialects.sqlite import insert
            case "postgres":
                from sqlalchemy.dialects.postgresql import insert
            case _:
                raise ValueError(f"Unsupported backend for bulk loading blocks on startup: {settings.backend.name}")

        statement = insert(cls).values(sql_blocks).on_conflict_do_nothing()
        db_session.execute(statement)
