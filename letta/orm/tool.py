import importlib
from inspect import getsource, isfunction
from types import ModuleType
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

# TODO everything in functions should live in this model
from letta.functions.schema_generator import generate_schema
from letta.orm.enums import ToolSourceType
from letta.orm.errors import NoResultFound
from letta.orm.mixins import OrganizationMixin, UserMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.tool import Tool as PydanticTool

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from letta.orm.organization import Organization
    from letta.orm.user import User


class Tool(SqlalchemyBase, OrganizationMixin, UserMixin):
    """Represents an available tool that the LLM can invoke.

    NOTE: polymorphic inheritance makes more sense here as a TODO. We want a superset of tools
    that are always available, and a subset scoped to the organization. Alternatively, we could use the apply_access_predicate to build
    more granular permissions.
    """

    __tablename__ = "tool"
    __pydantic_model__ = PydanticTool

    # Add unique constraint on (name, _organization_id)
    # An organization should not have multiple tools with the same name
    __table_args__ = (
        UniqueConstraint("name", "_organization_id", name="uix_name_organization"),
        UniqueConstraint("name", "_user_id", name="uix_name_user"),
    )

    name: Mapped[str] = mapped_column(doc="The display name of the tool.")
    description: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The description of the tool.")
    tags: Mapped[List] = mapped_column(JSON, doc="Metadata tags used to filter tools.")
    source_type: Mapped[ToolSourceType] = mapped_column(String, doc="The type of the source code.", default=ToolSourceType.json)
    source_code: Mapped[Optional[str]] = mapped_column(
        String, doc="The source code of the function if provided.", default=None, nullable=True
    )
    json_schema: Mapped[dict] = mapped_column(JSON, default=lambda: {}, doc="The OAI compatable JSON schema of the function.")
    module: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="the module path from which this tool was derived in the codebase."
    )

    # TODO: add terminal here eventually
    # This was an intentional decision by Sarah

    # relationships
    # TODO: Possibly add in user in the future
    # This will require some more thought and justification to add this in.
    user: Mapped["User"] = relationship("User", back_populates="tools", lazy="selectin")
    organization: Mapped["Organization"] = relationship("Organization", back_populates="tools", lazy="selectin")

    @classmethod
    def read_by_id(cls, db_session: "Session", id: str) -> "Tool":
        if found := db_session.query(cls).filter(cls.id == id, cls.is_deleted == False).scalar():
            return found
        raise NoResultFound(f"{cls.__name__} with id {id} not found")

    @classmethod
    def load_default_tools(cls, db_session: "Session", org: "Organization") -> None:
        """populates the db with default tools"""
        target_module = importlib.import_module("letta.functions.function_sets.base")
        functions_to_schema = cls._load_function_set(target_module)
        tags = ["base", "memgpt-base"]
        sql_tools = []
        for name, schema in functions_to_schema.items():
            source_code = getsource(schema["python_function"])
            sql_tools.append(
                cls(
                    name=name,
                    organization=org,
                    tags=tags,
                    source_type="python",
                    module=schema["module"],
                    source_code=source_code,
                    json_schema=schema["json_schema"],
                )
            )
        db_session.add_all(sql_tools)
        db_session.commit()

    @classmethod
    def _load_function_set(cls, target_module: ModuleType) -> dict:
        """Load the functions and generate schema for them, given a module object"""
        function_dict = {}

        for attr_name in dir(target_module):
            # Get the attribute
            attr = getattr(target_module, attr_name)

            # Check if it's a callable function and not a built-in or special method
            if isfunction(attr) and attr.__module__ == target_module.__name__:
                generated_schema = generate_schema(attr)
                function_dict[attr_name] = {
                    "module": getsource(target_module),
                    "python_function": attr,
                    "json_schema": generated_schema,
                }
        if not function_dict:
            raise ValueError(f"No functions found in target module {target_module}")
        return function_dict
