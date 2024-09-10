import importlib
from inspect import getsource, isfunction
from types import ModuleType
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

# TODO everything in functions should live in this model
from memgpt.functions.schema_generator import generate_schema
from memgpt.orm.enums import ToolSourceType
from memgpt.orm.errors import NoResultFound
from memgpt.orm.mixins import OrganizationMixin
from memgpt.orm.organization import Organization
from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.schemas.tool import Tool as PydanticTool

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class Tool(SqlalchemyBase, OrganizationMixin):
    """Represents an available tool that the LLM can invoke.

    NOTE: polymorphic inheritance makes more sense here as a TODO. We want a superset of tools
    that are always available, and a subset scoped to the organization. Alternatively, we could use the apply_access_predicate to build
    more granular permissions.
    """

    __tablename__ = "tool"
    __pydantic_model__ = PydanticTool

    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The display name of the tool.")
    # TODO: this needs to be a lookup table to have any value
    tags: Mapped[List] = mapped_column(JSON, doc="Metadata tags used to filter tools.")
    source_type: Mapped[ToolSourceType] = mapped_column(String, doc="The type of the source code.", default=ToolSourceType.json)
    source_code: Mapped[Optional[str]] = mapped_column(
        String, doc="The source code of the function if provided.", default=None, nullable=True
    )
    json_schema: Mapped[dict] = mapped_column(JSON, default=lambda: {}, doc="The OAI compatable JSON schema of the function.")
    module: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="the module path from which this tool was derived in the codebase."
    )

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="tools", lazy="selectin")

    @classmethod
    def read(cls, db_session:"Session", identifier:Optional[str] = None, name:Optional[str] = None) -> "Tool":
        if name:
            if found := db_session.query(cls).filter(cls.name == name, cls.is_deleted == False).scalar():
                return found
        elif identifier:
            return cls.read_by_id(db_session, identifier)
        raise NoResultFound(f"{cls.__name__} with name {name} not found")

    @classmethod
    def read_by_id(cls, db_session: "Session", id: str) -> "Tool":
        if found := db_session.query(cls).filter(cls.id == id, cls.is_deleted == False).scalar():
            return found
        raise NoResultFound(f"{cls.__name__} with id {id} not found")

    @classmethod
    def load_default_tools(cls, db_session: "Session") -> None:
        """populates the db with default tools"""
        target_module = importlib.import_module("memgpt.seeds.function_sets.base")
        functions_to_schema = cls._load_function_set(target_module)
        tags = ["base", "memgpt-base"]
        sql_tools = []
        org = Organization.default(db_session)
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
